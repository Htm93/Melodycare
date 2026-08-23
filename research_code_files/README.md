# MelodyCare

**An automated two-stage generative framework for converting Vietnamese commercial music into clinically-bounded therapeutic audio for individuals with Autism Spectrum Disorder (ASD).**

Final-year summer research project — Department of Computer Science and Engineering, Faculty of Engineering, The Chinese University of Hong Kong (CUHK AISTN Summer Research Internship, 2026).

**Author:** Hoang Tuan Minh
**Supervisor:** Nguyen Viet Anh (Assistant Professor)

---

## Overview

Individuals with ASD frequently experience auditory hyperacusis — ordinary commercial music (unpredictable dynamics, vocals, harsh transients, fast tempo) can trigger sensory overload instead of comfort. Generic "calming music" libraries aren't personalized to a specific song someone actually wants to hear, but that song's raw commercial mix is exactly the kind of unpredictable signal that causes overload.

MelodyCare addresses this with a **three-stage pipeline** that takes a specific Vietnamese commercial song and converts it into a version that preserves its melodic identity while enforcing seven objective, clinically-motivated acoustic-safety bounds (tempo, dynamic range, spectral roll-off, frequency balance, amplitude modulation, pitch intervals, and legato smoothness).

## Pipeline

```
Commercial song → Stage 1: DSP (vocal/drum removal) → Stage 2: Generative Model (AI) → Stage 3: DSP Safety Net → Therapeutic audio output
```

**Stage 1 — Deterministic DSP.** Demucs source separation (keep melody + bass, discard vocals/drums), BPM detection and correction, Butterworth spectral shaping, dynamic-range compression, legato smoothing, and a low-frequency "breathing" LFO. Builds every (input, target) training pair and is reused verbatim as Stage 3.

**Stage 2 — Conditioned Generative Model.** Fine-tunes `stabilityai/stable-audio-open-1.0` (24-layer diffusion transformer) with two new cross-attention conditioning modules — a CLAP-derived style vector projection and a novel input-audio projection that lets the model condition on a full source track despite the pretrained checkpoint having no native audio-to-audio pathway. Both modules are zero-initialized (ControlNet-style) and trained with a differential learning rate against the pretrained backbone to prevent catastrophic forgetting. Inference uses img2img/SDEdit-style partial-noise sampling to anchor output structure to the requested song.

**Stage 3 — Deterministic Compliance Safety Net.** Re-applies Stage 1's DSP corrections directly to Stage 2's generated output, so that clinical-safety compliance is never solely dependent on the generative model's statistical output.

## Technical Contributions

- **Duration-ratio-synchronized data cropping** — corrects a bug where independent random crops of input/target files paired unrelated time windows for ~95% of training pairs (BPM time-stretching changes duration).
- **Zero-init + differential learning rates** against catastrophic forgetting when fine-tuning a small (775-chunk) dataset.
- **Cross-attention conditioning for source audio** — works around the base model's fixed 64-channel input by reframing source-audio conditioning as variable-length cross-attention tokens instead of channel-concatenation.
- **img2img/SDEdit partial-noise sampling** for structural anchoring to the requested song's identity.
- **Vocal-removal preprocessing** to sidestep a diagnosed model weakness in vocal-to-instrumental translation, via deterministic upstream Demucs separation.

## Results

Evaluated on 10 in-sample and 10 genuinely unseen held-out songs across the seven clinical-safety checks:

| Check | Unseen (Stage 2) | Unseen (Stage 2+3) | In-sample (Stage 2) | In-sample (Stage 2+3) |
|---|---|---|---|---|
| BPM (60–80) | 10/10 | 10/10 | 10/10 | 10/10 |
| Spectral roll-off ≤5kHz | 10/10 | 10/10 | 10/10 | 10/10 |
| Dynamic range <13dB | 1/10 | **10/10** | 0/10 | **10/10** |
| Frequency boost 100–400Hz | 10/10 | 10/10 | 9/10 | 10/10 |
| LFO modulation | 10/10 | 10/10 | 10/10 | 10/10 |
| Pitch intervals ≤12st | 10/10 | 10/10 | 10/10 | 10/10 |
| Legato overlap 70–80% | 0/10 | 2/10 | 0/10 | 0/10 |
| **Overall (all 7)** | 0/10 | 2/10 | 0/10 | 0/10 |

The model generalizes cleanly on five of seven dimensions regardless of whether a song was in the training set. Stage 3's dynamic-range correction is reliable and population-independent. Legato-overlap correction remains the main open limitation, and an ablation study found the source-audio conditioning module's contribution could not be detected at the tested inference strength (0.65) — both reported honestly as scoped directions for future work rather than smoothed over. Full quantitative results (spectrogram comparisons, chroma/MFCC content-similarity, strength sweep, ablation study) are in the final report.

## Known Limitations

- Legato-overlap correction is inconsistent (transition-to-transition variance), not simply insufficient on average.
- The ablation study did not detect `InputAudioProjection`'s contribution at the tested strength (0.65) — a lower-strength re-test is a planned follow-up.
- Imperfect vocal isolation at inference time, dependent on Demucs' separation quality.
- The tested strength range (0.35–0.85) trends toward better compliance at the top end; higher values are untested.

## Repository Structure

| Area | Files |
|---|---|
| Config | `config.py`, `train_config.py`, `system_config.py`, `accelerate_config.yaml` |
| Data prep | `1_download_music.py`, `2_0_clean_file_name.py`, `2_1_BPM_filter.py`, `2_2_SampleRate_check.py`, `2_3_Filtering_files.py`, `2_generate_csv.py`, `3_source_separation.py`, `separate_chunks.py`, `separate_checked.py`, `slice_chunks.py`, `validate_dataset.py`, `find_missing.py`, `check_duration_mismatch.py` |
| Stage 1 DSP pipeline | `pipeline.py`, `analysis.py`, `dynamic.py`, `spectral.py`, `smoothing.py`, `drums.py`, `audio_utils.py` |
| Stage 2 model | `input_audio_projection.py`, `style_projection.py`, `model_utils.py`, `dataset.py`, `loader.py`, `collate.py`, `checkpoint.py`, `loss.py`, `loop.py`, `train.py` |
| Stage 3 / inference | `post_process.py`, `inference.py`, `demucs_preprocess.py` |
| Evaluation | `Check_result.py`, `check_result_file.py`, `compare_dsp_vs_ai.py`, `compare_music_content.py`, `evaluate_spectrograms.py`, `strength_sweep.py`, `analyze_strength_result.py`, `single_check.py`, `conversion_test.py`, `listen_to_result.py` |
| Sanity / debugging | `sanity_check.py`, `sanity_base_model.py`, `inspect_model.py`, `check_module_growth.py` |
| Misc | `main.py`, `worker.py`, `logging_utils.py`, `analyze_therapy_codition.py` |

## Dataset

**Not included in this repository.** The training set (775 paired 30-second chunks from 78 Vietnamese commercial songs) is built from copyrighted commercial music and cannot be redistributed. `melodycare_training_manifest.csv` is included for reference — it documents the manifest structure (input/target/style-vector paths, text prompts) but does not contain audio.

To reproduce training, you'll need to supply your own set of source songs and run the data-prep pipeline (`1_download_music.py` → `3_source_separation.py`) to regenerate the manifest.

## Setup

> **TODO:** add a `requirements.txt` / environment file with pinned versions before publishing.

Key dependencies used in this project:
- PyTorch, HuggingFace Accelerate (multi-GPU DDP)
- `stable_audio_tools` / `stabilityai/stable-audio-open-1.0`
- `bitsandbytes` (8-bit AdamW)
- Demucs (`htdemucs`)
- CLAP (for style-vector extraction)
- librosa

## Usage

Single-GPU training:
```bash
python train.py
```

Multi-GPU training (Accelerate):
```bash
accelerate launch --config_file configs/accelerate_config.yaml train.py
```

See each script's docstring/`--help` for full CLI arguments (e.g. `inference.py --ablate_input_audio` for the ablation study, `strength_sweep.py` for the strength-parameter sweep).

## License

> **TODO:** add a LICENSE file (e.g. MIT or Apache 2.0).

## Acknowledgments

Completed as part of the CUHK Faculty of Engineering Undergraduate Summer Research Internship (AISTN), under the supervision of Prof. Nguyen Viet Anh.
