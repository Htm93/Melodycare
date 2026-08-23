"""
MelodyCare Inference — Generate Therapeutic Audio from a Checkpoint

Loads a trained checkpoint (DiT + style_proj + input_audio_proj, as saved by
training/checkpoint.py) and runs it on a source input audio file to produce
a therapeutic-style output.

This is the ONE inference script — normal generation and the source-audio
ablation diagnostic are both handled here via the --ablate_input_audio flag,
not separate files. Keeping a single script avoids exactly the kind of
feature drift that happened when a second copy was made.

Sampling approach:
    diffusion_objective == "v" (continuous-time, confirmed via runtime
    inspection — see conversation history / training/loss.py header).
    This script implements a simple deterministic Euler sampler using the
    same get_alphas_sigmas() schedule the model was trained with. It is NOT
    using stable_audio_tools' own sampling utilities (we haven't verified
    their exact call signature), so treat this as a straightforward,
    correct-but-basic baseline sampler — good enough to judge checkpoint
    quality, not necessarily the fastest or highest-fidelity option
    available.

Usage (normal generation):
    python /path/to/training_output/check_inference/inference.py \
        --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-7399 \
        --input_audio    /path/to/dataset/raw/Tier_2_Acceptable_81-100_BPM_95/buong_bui_anh_tuan_lyric_video_d.wav \
        --output_audio   /path/to/training_output/check_inference/normal.wav \
        --num_steps      50 \
        --seed           42

Usage (ablation A/B test — same seed, add --ablate_input_audio):
    python .../inference.py \
        --checkpoint_dir .../checkpoint-7399 \
        --input_audio    .../buong_bui_anh_tuan_lyric_video_d.wav \
        --output_audio   .../ablated.wav \
        --num_steps      50 \
        --seed           42 \
        --ablate_input_audio

Usage (img2img — anchor output to actual input structure, NEW):
    python .../inference.py \
        --checkpoint_dir .../checkpoint-2450 \
        --input_audio    .../buong_bui_anh_tuan_lyric_video_d.wav \
        --output_audio   .../img2img_test.wav \
        --num_steps      50 \
        --strength       0.65
    # Lower --strength (e.g. 0.4) = closer to input, less transformation.
    # Higher --strength (e.g. 0.9) = more creative, closer to old pure-noise
    # behavior. --strength 1.0 = exactly the old behavior (pure noise start).

    python /path/to/training_output/check_inference/inference.py \
        --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-7410 \
        --input_audio    /path/to/dataset/Karaoke_HD_Bung_Full_Beat_Gc_Bi_Anh_Tun_Newtitan.wav \
        --output_audio /path/to/training_output/check_inference/strength_sweep/str40.wav \
        --seed 42 \
        --strength 0.40
"""
import argparse
import os
import sys

# =============================================================================
# Dynamic System Path Resolution
# =============================================================================
# Resolves project root (/path/to/training_dir) from nested script:
# training_result/check_inference/inference.py (2 levels up). Needed because
# Python only auto-adds the SCRIPT's own directory to sys.path, not the
# working directory — without this, `import config` / `from models.loader
# import ...` fail when this script lives in a subfolder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import soundfile as sf
import numpy as np
import librosa
from safetensors.torch import load_file as load_safetensors

# models/loader.py logs via utils.logging_utils.get_logger, which wraps
# accelerate's logger — that raises RuntimeError unless Accelerate's
# internal state has been initialized first. training/loop.py does this
# implicitly via Accelerator(...); for single-process inference we just
# need the lightweight PartialState() instead of a full Accelerator.
from accelerate.state import PartialState
PartialState()

import config
from models.loader import load_stable_audio_model
from stable_audio_tools.inference.sampling import get_alphas_sigmas


def detect_duration_ratio(waveform: torch.Tensor, sr: int, target_bpm: float = 60.0) -> float:
    """
    Detect the input song's BPM and derive the same duration_ratio the
    training data was built with (see analysis.py's enforce_bpm and
    dataset.py's ratio-synchronized cropping):

        - If BPM already in the 60-80 therapy range: ratio = 1.0 (no stretch,
          matches DSP pipeline's is_bpm_valid=True branch, which skips
          enforce_bpm entirely).
        - Otherwise: ratio = current_bpm / target_bpm, matching the DSP's
          librosa.effects.time_stretch(rate=target_bpm/current_bpm), whose
          OUTPUT duration = input_duration / rate = input_duration *
          (current_bpm/target_bpm).

    This is what tells the model how much output audio (in real time) it
    should be generating for a given amount of input audio.
    """
    mono = waveform.mean(dim=0).numpy()
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    current_bpm = float(np.squeeze(tempo))

    if 60.0 <= current_bpm <= 80.0:
        ratio = 1.0
    else:
        ratio = current_bpm / target_bpm

    print(f"Detected input BPM: {current_bpm:.1f} -> duration_ratio: {ratio:.3f}")
    return ratio


def build_therapy_prompt() -> str:
    """Same prompt construction as 2_generate_csv.py, duplicated here since
    that file's name isn't a valid Python module identifier to import from."""
    return (
        f"A slow, calm instrumental therapeutic music track at exactly "
        f"{config.THERAPY_BPM_TARGET} BPM, strictly within the "
        f"{config.THERAPY_BPM_MIN} to {config.THERAPY_BPM_MAX} BPM therapeutic range. "
        f"The rhythm is steady, predictable, and metronomic. "
        f"No tempo variations, rubato, or rhythmic surprises. "
        f"The overall dynamic range is extremely compressed, "
        f"with a peak-to-floor variance of less than {config.THERAPY_DYNAMIC_DB:.0f} dB. "
        f"Volume is consistent with no sudden loud transients or abrupt silences. "
        f"Notes transition smoothly with legato style, overlapping between "
        f"{config.THERAPY_LEGATO_MIN}% and {config.THERAPY_LEGATO_MAX}% of their duration. "
        f"No staccato articulations. Each note sustains gently into the next. "
        f"A subtle breathing quality is present with amplitude modulation oscillating "
        f"between {config.THERAPY_LFO_MIN_HZ} Hz and {config.THERAPY_LFO_MAX_HZ} Hz. "
        f"The frequency spectrum rolls off sharply above {config.THERAPY_ROLLOFF_KHZ} kHz. "
        f"No harsh high-frequency components above {config.THERAPY_ROLLOFF_KHZ} kHz. "
        f"Sub-bass frequencies below {config.THERAPY_SUB_CUTOFF_HZ} Hz are absent. "
        f"Compared to the baseline therapeutic music reference, there is enhanced warmth "
        f"in the {config.THERAPY_BOOST_LOW_HZ} Hz to {config.THERAPY_BOOST_HIGH_HZ} Hz band. "
        f"Minimalist, sparse note density with long durations and generous musical space. "
        f"Deeply calming, safe, and predictable — designed for sensory regulation "
        f"in individuals with Autism Spectrum Disorder."
    )


def load_checkpoint_weights(checkpoint_dir, dit, style_proj, input_audio_proj):
    """
    Load the three trainable modules' weights directly from the safetensors/
    .pt files saved by training/checkpoint.py — no Accelerator needed for
    single-process inference.

    Order matches accelerator.prepare(model.model, style_proj,
    input_audio_proj, ...) in training/loop.py, so:
        model.safetensors   -> dit
        model_1.safetensors -> style_proj
        model_2.safetensors -> input_audio_proj
    """
    dit_path = os.path.join(checkpoint_dir, "model.safetensors")
    dit.load_state_dict(load_safetensors(dit_path))
    print(f"[OK] DiT weights loaded from {dit_path}")

    style_pt = os.path.join(checkpoint_dir, "style_projection.pt")
    if os.path.exists(style_pt):
        style_proj.load_state_dict(torch.load(style_pt, map_location="cpu"))
    else:
        style_proj.load_state_dict(load_safetensors(os.path.join(checkpoint_dir, "model_1.safetensors")))
    print("[OK] style_proj weights loaded")

    input_audio_pt = os.path.join(checkpoint_dir, "input_audio_projection.pt")
    if os.path.exists(input_audio_pt):
        input_audio_proj.load_state_dict(torch.load(input_audio_pt, map_location="cpu"))
    else:
        input_audio_proj.load_state_dict(load_safetensors(os.path.join(checkpoint_dir, "model_2.safetensors")))
    print("[OK] input_audio_proj weights loaded")


@torch.no_grad()
def build_conditioning(model, style_proj, input_audio_proj, text_prompt, input_latents, device, chunk_duration_s,
                        ablate_input_audio: bool = False):
    """
    Mirrors the conditioning-assembly logic in training/loop.py exactly —
    prompt + seconds_start/seconds_total -> conditioner -> get_conditioning_inputs,
    then splice on style + input-audio tokens.

    ablate_input_audio: if True, force input_audio_tokens to zero — used as
    an A/B diagnostic to test whether the model is actually using the source
    audio conditioning. If output sounds/measures nearly identical with vs
    without this flag, the model isn't meaningfully using source content yet.
    """
    cond_input = [{
        "prompt"        : text_prompt,
        "seconds_start" : 0,
        "seconds_total" : chunk_duration_s,
    }]
    conditioning_tensors = model.conditioner(cond_input, device=device)
    base_cond = model.get_conditioning_inputs(conditioning_tensors)

    text_embeddings = base_cond["cross_attn_cond"]
    text_mask       = base_cond.get("cross_attn_mask", base_cond.get("cross_attn_cond_mask"))

    style_vector_placeholder = torch.zeros(1, config.STYLE_VECTOR_DIM, device=device)
    style_vector_path = os.path.join(config.STYLE_VECTOR_DIR, f"{config.STYLE_VECTOR_STEM}.pt")
    if os.path.exists(style_vector_path):
        style_vector_placeholder = torch.load(style_vector_path, map_location=device).float().unsqueeze(0)
    else:
        print(f"[WARN] style vector not found at {style_vector_path} — using zeros")

    style_embeddings   = style_proj(style_vector_placeholder)
    input_audio_tokens = input_audio_proj(input_latents)
    if ablate_input_audio:
        input_audio_tokens = torch.zeros_like(input_audio_tokens)

    combined_cond = torch.cat([text_embeddings, style_embeddings, input_audio_tokens], dim=1)

    style_mask = torch.ones((1, 1), device=device, dtype=text_mask.dtype)
    input_audio_mask = torch.ones(input_audio_tokens.shape[:2], device=device, dtype=text_mask.dtype)
    combined_mask = torch.cat([text_mask, style_mask, input_audio_mask], dim=1)

    conditioning = dict(base_cond)
    conditioning["cross_attn_cond"] = combined_cond
    conditioning["cross_attn_mask"] = combined_mask
    conditioning.pop("cross_attn_cond_mask", None)
    return conditioning


@torch.no_grad()
def sample_v_objective(dit, shape, conditioning, device, num_steps=50,
                        init_latents=None, strength=1.0):
    """
    Deterministic Euler sampler for a v-objective diffusion model.

    img2img / SDEdit mode (init_latents provided, strength < 1.0):
        Instead of starting from pure random noise (strength=1.0, the OLD
        default — this generates from scratch, only indirectly influenced by
        input via cross-attention conditioning), starts from the ACTUAL
        input audio's own encoded latents with PARTIAL noise added, then
        denoises from there. This structurally anchors the output to the
        real input content from step one, rather than relying entirely on
        the model having learned to attend to conditioning tokens.

        No retraining was needed for this — during training, t is sampled
        continuously and uniformly across [0,1], so the model already
        learned to denoise from ANY noise level, not just t=1. We're simply
        choosing to start the inference-time sampling loop partway through
        that range instead of at the very top.

    strength: 0.0-1.0, matches the familiar Stable Diffusion img2img
        convention. 1.0 = pure noise start (old behavior, ignores input
        structure, output driven by conditioning/prompt only). Lower values
        = start closer to the actual input (more structurally preserved,
        less "creative" transformation room for the model). A moderate
        value (~0.5-0.7) is the reasonable starting point: enough noise for
        the model to meaningfully transform the audio's tempo/dynamics/
        spectral character, while the surviving structure still anchors the
        output to the real input content.
    """
    t_start = strength

    if init_latents is not None and t_start < 1.0:
        noise = torch.randn_like(init_latents)
        alpha0, sigma0 = get_alphas_sigmas(torch.tensor([t_start], device=device).expand(shape[0]))
        alpha0 = alpha0[:, None, None]
        sigma0 = sigma0[:, None, None]
        x = init_latents * alpha0 + noise * sigma0
    else:
        x = torch.randn(shape, device=device)

    timesteps = torch.linspace(t_start, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur  = timesteps[i]
        t_next = timesteps[i + 1]

        t_batch = t_cur.expand(shape[0])
        alpha, sigma = get_alphas_sigmas(t_batch)
        alpha = alpha[:, None, None]
        sigma = sigma[:, None, None]

        v_pred = dit(
            x, t_batch,
            cross_attn_cond   = conditioning.get("cross_attn_cond"),
            cross_attn_mask   = conditioning.get("cross_attn_mask"),
            global_cond       = conditioning.get("global_cond"),
            prepend_cond      = conditioning.get("prepend_cond"),
            prepend_cond_mask = conditioning.get("prepend_cond_mask"),
            cfg_dropout_prob  = 0.0,
        )

        pred_x0    = x * alpha - v_pred * sigma
        pred_noise = x * sigma + v_pred * alpha

        alpha_next, sigma_next = get_alphas_sigmas(t_next.expand(shape[0]))
        alpha_next = alpha_next[:, None, None]
        sigma_next = sigma_next[:, None, None]

        x = pred_x0 * alpha_next + pred_noise * sigma_next

    return x


def extract_fixed_window(waveform: torch.Tensor, start_frame: int, num_frames: int, chunk_len: int) -> torch.Tensor:
    """
    Extract [start_frame : start_frame+num_frames] from waveform, then
    zero-pad or crop to exactly chunk_len samples. Mirrors dataset.py's
    behavior: for a stretched song, the input segment covering one 30s
    output chunk is SHORTER than chunk_len (ratio > 1), so it gets
    zero-padded to fill the window — same signal the model was trained on
    for CONDITIONING purposes.
    """
    start_frame = max(0, start_frame)
    end_frame   = min(waveform.shape[1], start_frame + max(1, num_frames))
    seg = waveform[:, start_frame:end_frame]
    if seg.shape[1] < chunk_len:
        seg = torch.nn.functional.pad(seg, (0, chunk_len - seg.shape[1]))
    else:
        seg = seg[:, :chunk_len]
    return seg


def extract_stretched_window(waveform: torch.Tensor, start_frame: int, num_frames: int, chunk_len: int) -> torch.Tensor:
    """
    Like extract_fixed_window, but STRETCHES (resamples) the real content to
    fill the entire chunk_len window instead of zero-padding it.

    WHY THIS EXISTS: for img2img (see sample_v_objective's init_latents),
    using the zero-padded version as the DENOISING STARTING POINT causes
    large dead-silence stretches in the output — with partial noise
    (strength < 1.0), the near-silent padded region doesn't get enough
    randomness to let the model generate real content there, since training
    never saw a "mostly silent + a little noise" starting point (training
    always denoised from a fully clean, continuously-voiced TARGET, never
    from a padded/silent x). Conditioning tokens still use the zero-padded
    version elsewhere (that matches what training actually used there) —
    this stretched version is ONLY for the img2img starting latent.
    """
    start_frame = max(0, start_frame)
    end_frame   = min(waveform.shape[1], start_frame + max(1, num_frames))
    real_segment = waveform[:, start_frame:end_frame]

    if real_segment.shape[1] == 0:
        return torch.zeros(waveform.shape[0], chunk_len)

    if real_segment.shape[1] >= chunk_len:
        return real_segment[:, :chunk_len]

    import torchaudio
    stretched = torchaudio.functional.resample(
        real_segment, orig_freq=real_segment.shape[1], new_freq=chunk_len
    )
    if stretched.shape[1] < chunk_len:
        stretched = torch.nn.functional.pad(stretched, (0, chunk_len - stretched.shape[1]))
    else:
        stretched = stretched[:, :chunk_len]
    return stretched


def crossfade_stitch(chunks: list, overlap_samples: int) -> torch.Tensor:
    """
    Overlap-add stitch a list of (2, chunk_len) chunk waveforms into one
    continuous track, linearly crossfading over `overlap_samples` at each
    boundary.
    """
    if len(chunks) == 1:
        return chunks[0]

    device  = chunks[0].device
    fade_in  = torch.linspace(0, 1, overlap_samples, device=device)
    fade_out = 1.0 - fade_in

    chunk_len   = chunks[0].shape[1]
    hop         = chunk_len - overlap_samples
    total_len   = hop * (len(chunks) - 1) + chunk_len

    output = torch.zeros(2, total_len, device=device)
    for i, chunk in enumerate(chunks):
        start = i * hop
        if i == 0:
            output[:, start:start + chunk_len] += chunk
        else:
            output[:, start:start + overlap_samples] *= fade_out
            chunk = chunk.clone()
            chunk[:, :overlap_samples] *= fade_in
            output[:, start:start + chunk_len] += chunk

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--input_audio",    type=str, required=True)
    parser.add_argument("--output_audio",   type=str, required=True)
    parser.add_argument("--num_steps",      type=int, default=50)
    parser.add_argument("--text_prompt",    type=str, default=None,
                         help="Override the default therapy prompt")
    parser.add_argument("--overlap_seconds", type=float, default=2.0,
                         help="Overlap between consecutive 30s chunks, crossfaded at stitch time")
    parser.add_argument("--strength", type=float, default=0.65,
                         help="img2img denoising strength, 0.0-1.0 (Stable Diffusion convention). "
                              "1.0 = pure random noise start (old behavior — generates from scratch, "
                              "ignores input structure). Lower = starts closer to the actual input "
                              "audio's own latents, more structurally preserved. Default 0.65 is a "
                              "moderate starting point — reduce toward ~0.4 for more input fidelity, "
                              "increase toward ~0.9 for more creative transformation.")
    parser.add_argument("--ablate_input_audio", action="store_true",
                         help="Diagnostic: zero out input-audio conditioning tokens to test "
                              "whether the model is actually using source-audio content")
    parser.add_argument("--seed", type=int, default=42,
                         help="Fixes the starting noise so runs are directly comparable — "
                              "e.g. normal vs --ablate_input_audio should use the SAME seed "
                              "so any difference in output is from the conditioning, not from "
                              "different random noise each run")
    parser.add_argument("--duration_ratio", type=float, default=None,
                         help="Override the auto-detected duration_ratio instead of running BPM "
                              "detection on --input_audio directly. IMPORTANT if using a vocal/drum-"
                              "removed input (e.g. from demucs_preprocess.py): training data's ground "
                              "truth BPM was always detected from the FULL mix (drums give beat-"
                              "tracking its strongest signal), so detecting BPM from a stripped "
                              "instrumental can be less accurate AND methodologically inconsistent "
                              "with how the model's training ratios were established. Pass the ratio "
                              "printed by running detect_duration_ratio on the ORIGINAL full-mix file "
                              "instead of letting this script re-detect it from the stripped audio.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Using seed: {args.seed}")
    if args.ablate_input_audio:
        print("*** ABLATION MODE: input-audio conditioning tokens forced to zero ***")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pretrained model + projections...")
    model, style_proj, input_audio_proj, model_config = load_stable_audio_model(
        model_id               = config.PRETRAINED_MODEL_ID,
        device                 = device,
        gradient_checkpointing = False,
    )
    dit = model.model
    dit.eval()
    style_proj.eval()
    input_audio_proj.eval()

    print(f"Loading checkpoint weights from {args.checkpoint_dir} ...")
    load_checkpoint_weights(args.checkpoint_dir, dit, style_proj, input_audio_proj)

    print(f"Loading input audio: {args.input_audio}")
    import torchaudio
    waveform, sr = torchaudio.load(args.input_audio)
    if sr != config.SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    orig_len          = waveform.shape[1]
    input_duration_s  = orig_len / config.SAMPLE_RATE
    chunk_len         = config.SAMPLE_RATE * config.CHUNK_DURATION_S
    chunk_duration_s  = float(config.CHUNK_DURATION_S)
    overlap_s         = args.overlap_seconds

    if args.duration_ratio is not None:
        duration_ratio = args.duration_ratio
        print(f"Using provided duration_ratio: {duration_ratio:.3f} (skipping auto BPM detection)")
    else:
        duration_ratio = detect_duration_ratio(waveform, config.SAMPLE_RATE)
    output_duration_s = input_duration_s * duration_ratio
    print(f"Input duration: {input_duration_s:.1f}s -> expected output duration: {output_duration_s:.1f}s")

    hop_output_s = max(1.0, chunk_duration_s - overlap_s)
    if output_duration_s <= chunk_duration_s:
        output_starts_s = [0.0]
    else:
        output_starts_s = list(np.arange(0.0, output_duration_s - chunk_duration_s + 1e-6, hop_output_s))
        if output_starts_s[-1] + chunk_duration_s < output_duration_s:
            output_starts_s.append(output_duration_s - chunk_duration_s)

    text_prompt = args.text_prompt or build_therapy_prompt()

    output_chunks = []
    for i, out_start_s in enumerate(output_starts_s):
        in_start_s    = out_start_s / duration_ratio
        in_duration_s = chunk_duration_s / duration_ratio
        in_start_frame  = int(round(in_start_s * config.SAMPLE_RATE))
        in_num_frames   = int(round(in_duration_s * config.SAMPLE_RATE))

        print(f"Chunk {i+1}/{len(output_starts_s)}  "
              f"output[{out_start_s:.1f}s..{out_start_s+chunk_duration_s:.1f}s]  "
              f"<- input[{in_start_s:.1f}s..{in_start_s+in_duration_s:.1f}s]")

        chunk_wave = extract_fixed_window(waveform, in_start_frame, in_num_frames, chunk_len)
        chunk_wave = chunk_wave.unsqueeze(0).to(device)

        with torch.no_grad():
            input_latents = model.pretransform.encode(chunk_wave)

        conditioning = build_conditioning(
            model, style_proj, input_audio_proj,
            text_prompt, input_latents, device,
            config.CHUNK_DURATION_S,
            ablate_input_audio=args.ablate_input_audio,
        )

        # img2img starting point: stretched (not zero-padded) version, to
        # avoid feeding the diffusion process a mostly-silent starting
        # latent — see extract_stretched_window's docstring for why.
        if args.strength < 1.0:
            stretched_wave = extract_stretched_window(waveform, in_start_frame, in_num_frames, chunk_len)
            stretched_wave = stretched_wave.unsqueeze(0).to(device)
            with torch.no_grad():
                img2img_source_latents = model.pretransform.encode(stretched_wave)
        else:
            img2img_source_latents = None  # strength=1.0 -> pure noise, no img2img source needed

        output_latents = sample_v_objective(
            dit, input_latents.shape, conditioning, device, num_steps=args.num_steps,
            init_latents=img2img_source_latents, strength=args.strength,
        )

        with torch.no_grad():
            chunk_output = model.pretransform.decode(output_latents)

        output_chunks.append(chunk_output.squeeze(0).clamp(-1.0, 1.0).cpu())

    overlap_samples = int(overlap_s * config.SAMPLE_RATE)
    if len(output_chunks) == 1:
        target_len  = int(round(output_duration_s * config.SAMPLE_RATE))
        final_audio = output_chunks[0][:, :max(1, target_len)]
    else:
        final_audio = crossfade_stitch(output_chunks, overlap_samples)
        target_len  = int(round(output_duration_s * config.SAMPLE_RATE))
        final_audio = final_audio[:, :target_len]

    os.makedirs(os.path.dirname(args.output_audio) or ".", exist_ok=True)
    # Explicit PCM_16 subtype: soundfile defaults to 32-bit float WAV for
    # float32 arrays, which many simple audio players (including VS Code's
    # built-in preview) can't play even though the file is perfectly valid.
    # Standard 16-bit PCM is broadly compatible and the quality loss is
    # imperceptible for listening/QA purposes.
    sf.write(args.output_audio, final_audio.T.numpy(), config.SAMPLE_RATE, subtype="PCM_16")
    print(f"[DONE] Saved: {args.output_audio} ({final_audio.shape[1] / config.SAMPLE_RATE:.1f}s, "
          f"duration_ratio={duration_ratio:.3f}, strength={args.strength:.2f})")


if __name__ == "__main__":
    main()