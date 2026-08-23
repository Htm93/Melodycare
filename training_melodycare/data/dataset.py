"""
MelodyCare PyTorch Dataset (v2)

v2 UPDATE:
    - Loads pre-mixed input/target files directly (no stem mixing)
    - Uses disk-efficient load_audio() with frame_offset
    - Simplified CSV schema: input_audio, target_audio columns only
    - Removed: _load_and_mix(), mix_stems(), stem path handling
"""
import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset

import config
from data.audio_utils import load_audio, to_stereo, apply_gain_augmentation
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class MelodyCareDataset(Dataset):
    """
    Dataset for MelodyCare audio-to-audio therapeutic translation.

    Each sample provides:
        input_audio  : (2, chunk_len) float32 — original commercial audio
        target_audio : (2, chunk_len) float32 — DSP-processed therapeutic audio
        style_vector : (512,) float32          — CLAP style conditioning vector
        text_prompt  : str                     — therapy text prompt
        song_name    : str                     — identifier for logging

    CSV schema expected (v2):
        song_name, input_audio, target_audio,
        style_vector_pt, style_vector_npy,
        text_prompt, negative_prompt

    Args:
        csv_path        : path to master training CSV
        sample_rate     : target sample rate (default: config.SAMPLE_RATE)
        chunk_duration_s: audio chunk in seconds (default: config.CHUNK_DURATION_S)
        augment         : enable gain augmentation (True for train, False for eval)
    """

    # v2: simplified required columns — no stem paths
    REQUIRED_FILE_COLS = [
        "input_audio",
        "target_audio",
        "style_vector_pt",
    ]

    def __init__(
        self,
        csv_path        : str   = config.CSV_OUTPUT_PATH,
        sample_rate     : int   = config.SAMPLE_RATE,
        chunk_duration_s: int   = config.CHUNK_DURATION_S,
        augment         : bool  = True,
    ):
        self.sample_rate = sample_rate
        self.chunk_len   = sample_rate * chunk_duration_s
        self.augment     = augment
        self.records     = self._parse_csv(csv_path)

        logger.info(
            f"Dataset (v2): {len(self.records)} pre-mixed pairs loaded "
            f"| chunk={chunk_duration_s}s @ {sample_rate}Hz "
            f"| augment={augment}"
        )

    def _parse_csv(self, csv_path: str) -> list[dict]:
        """
        Parse CSV and validate file existence.
        Skips rows with missing files and logs warnings.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                missing = [
                    col for col in self.REQUIRED_FILE_COLS
                    if not os.path.exists(row.get(col, ""))
                ]
                if missing:
                    logger.warning(
                        f"Row {i} ({row.get('song_name', '?')}): "
                        f"missing files {missing} — skipping"
                    )
                    continue
                records.append(row)

        if not records:
            raise ValueError(f"No valid records found in {csv_path}")

        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_style_vector(self, path: str) -> torch.Tensor:
        """Load pre-computed CLAP style vector from .pt file."""
        return torch.load(path, map_location="cpu").float()   # (512,)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # ── Load input and target audio, TIME-SYNCHRONIZED ─────────────
        # Many song pairs in this dataset went through BPM time-stretching
        # in the DSP pipeline (analysis.py enforce_bpm), so the target file
        # can be a very different duration than the input file for the same
        # musical content (e.g. input 272s -> target 389s, ratio 1.43x —
        # confirmed via check_duration_mismatch.py: this affects ~95% of
        # pairs in this dataset, not an edge case).
        #
        # Loading input_audio and target_audio with independent random crops
        # (the old approach) picks two UNRELATED time windows almost every
        # time, regardless of stretching — silently pairing wrong content
        # for the vast majority of training samples. Fix: pick one crop
        # position in the TARGET's timeline, then derive the corresponding
        # INPUT crop position/duration by scaling with the measured
        # input/target duration ratio for this specific pair.
        random_crop = self.augment

        target_info = torchaudio.info(rec["target_audio"])
        input_info  = torchaudio.info(rec["input_audio"])

        target_duration_s = target_info.num_frames / target_info.sample_rate
        input_duration_s  = input_info.num_frames / input_info.sample_rate
        # Empirical per-pair stretch ratio, derived directly from file
        # durations — no dependency on DSP-pipeline metadata. ratio > 1
        # means target runs longer than input (tempo was slowed down).
        ratio = target_duration_s / max(input_duration_s, 1e-6)

        # ── Pick the target crop window (this defines the canonical 30s
        # output segment) ──
        chunk_duration_s = self.chunk_len / self.sample_rate
        max_target_start = max(0.0, target_duration_s - chunk_duration_s)
        if random_crop:
            target_start_s = torch.empty(1).uniform_(0, max_target_start).item() if max_target_start > 0 else 0.0
        else:
            target_start_s = max_target_start / 2.0

        target_frame_offset = int(target_start_s * target_info.sample_rate)
        target_native_frames = int(chunk_duration_s * target_info.sample_rate)

        # ── Derive the corresponding input crop window ──────────────────
        # Same musical passage, scaled into input's (untouched) timeline.
        input_start_s    = target_start_s / ratio
        input_duration_here_s = chunk_duration_s / ratio
        input_frame_offset  = int(input_start_s * input_info.sample_rate)
        input_native_frames = int(input_duration_here_s * input_info.sample_rate)

        target_audio = load_audio(
            path        = rec["target_audio"],
            target_sr   = self.sample_rate,
            chunk_len   = self.chunk_len,
            frame_offset_override      = target_frame_offset,
            native_num_frames_override = target_native_frames,
        )   # (chunk_len,) mono

        input_audio = load_audio(
            path        = rec["input_audio"],
            target_sr   = self.sample_rate,
            chunk_len   = self.chunk_len,
            frame_offset_override      = input_frame_offset,
            native_num_frames_override = input_native_frames,
        )   # (chunk_len,) mono — zero-padded if input_duration_here_s < chunk_duration_s
            # (i.e. ratio > 1 / song was slowed down), which is the common
            # case here. This is intentional: the model sees the ORIGINAL,
            # un-stretched input content (silence-padded to fill the fixed
            # window) and must learn to produce the full-length, tempo-
            # stretched target itself — matching the stated goal of the
            # model learning to time-stretch rather than relying on a
            # separate pre-processing step at inference.

        # ── Gain augmentation ─────────────────────────────────────────
        # Apply identical gain to both input and target so the model
        # does not learn to change amplitude between input and output
        if self.augment:
            gain_db     = torch.FloatTensor(1).uniform_(-3.0, 3.0)
            gain_linear = 10.0 ** (gain_db / 20.0)
            input_audio  = (input_audio  * gain_linear).clamp(-1.0, 1.0)
            target_audio = (target_audio * gain_linear).clamp(-1.0, 1.0)

        # ── Convert mono → stereo ─────────────────────────────────────
        # Stable Audio 3 expects (channels, samples) with channels=2
        input_audio  = to_stereo(input_audio)    # (2, chunk_len)
        target_audio = to_stereo(target_audio)   # (2, chunk_len)

        # ── Style vector ──────────────────────────────────────────────
        style_vector = self._load_style_vector(rec["style_vector_pt"])

        return {
            "input_audio" : input_audio.float(),    # (2, chunk_len)
            "target_audio": target_audio.float(),   # (2, chunk_len)
            "style_vector": style_vector,           # (512,)
            "text_prompt" : rec.get("text_prompt", ""),
            "song_name"   : rec.get("song_name", "unknown"),
        }