# Smart and Intelligent Video Analysis System using Agentic AI

An AI-powered surveillance video quality analysis system that uses a Vision-Language Model (VLM) combined with an agentic decision pipeline to automatically detect, diagnose, and correct technical quality defects in video footage — deployed as an interactive Streamlit app.

Rather than just flagging bad footage, the system closes the loop: **detect → decide → act.** A VLM diagnoses each frame, a decision agent reads the structured output, and a GAN-based enhancement service automatically restores frames that need it — no manual intervention required.

## Why this exists

Surveillance footage is frequently degraded by blur, sensor noise, compression artifacts, and poor lighting, which undermines its usefulness for monitoring, review, or downstream automated analysis. Manual review is slow and inconsistent, and naive rule-based quality checks are brittle. This project uses a VLM as a constrained, privacy-conscious quality auditor — evaluating *only* technical defects, never scene content — and pairs it with an autonomous correction step.

## Architecture

```
Video Upload
     │
     ▼
Frame Sampling (configurable interval) + Metadata Extraction
     │
     ▼
┌─────────────────────────────────────────────┐
│  Perception Stage                            │
│  Qwen3-VL-2B (4-bit NF4 quantized, ~2.5GB)    │
│  Prompt-constrained to technical-quality-only │
└─────────────────────────────────────────────┘
     │
     ▼
Structuring Stage — regex parsing → clean JSON
     │
     ▼
Aggregation Stage — per-frame verdicts → video-level summary
     │
     ▼
┌─────────────────────────────────────────────┐
│  Decision Agent (decision_agent.py)           │
│  Reads JSON, checks severity thresholds       │
└─────────────────────────────────────────────┘
     │
     ▼ (if flagged)
Action Stage — Real-ESRGAN 4x super-resolution enhancement
     │
     ▼
Enhanced frame + reasoning surfaced in UI
```

## Features

- **VLM-based quality diagnosis** — evaluates blur (motion/defocus), sensor noise, compression artifacts (macroblocking, banding, pixelation), and lighting (over/underexposure, low contrast), each with presence, confidence, severity, and evidence.
- **Privacy-conscious by design** — system prompts strictly forbid the VLM from describing scene content (people, vehicles, objects), constraining it to technical quality only.
- **Structured, machine-readable output** — free-text VLM responses are parsed into clean JSON via regex extraction, not left as unstructured text.
- **Autonomous correction loop** — a decision agent reads each frame's JSON verdict and automatically triggers Real-ESRGAN 4x super-resolution enhancement when blur, noise, or compression severity crosses a threshold (lighting defects are excluded, since super-resolution doesn't address exposure issues).
- **Offline-capable** — includes an alternate Moondream2 VLM backend for fully local, no-cloud-dependency inference.
- **Synthetic defect injection** — a built-in module (`generate_defects.py`) applies controllable blur/noise/compression/lighting degradation at adjustable severity, producing labeled test videos to validate detection accuracy.
- **4-bit quantized inference** — runs the VLM in NF4 precision, keeping VRAM usage under ~2.5GB for practical single-GPU deployment.

## Tech Stack

| Component | Technology |
|---|---|
| VLM (primary) | Qwen3-VL-2B-Instruct (4-bit NF4 quantized) |
| VLM (offline fallback) | Moondream2 |
| Quantization | `bitsandbytes` |
| Enhancement | Real-ESRGAN (GAN-based 4x super-resolution) |
| Frame/video processing | OpenCV |
| ML framework | PyTorch, Transformers |
| Web interface | Streamlit |

## Project Structure

```
├── app.py                     # Streamlit app — main pipeline orchestration
├── config.py                  # Paths, model IDs, thresholds
├── prompts.py                 # VLM system prompts (quality-only constraint)
├── decision_agent.py           # Reads VLM JSON, decides on enhancement action
├── enhancement_service.py      # Real-ESRGAN GAN-based frame enhancement
├── video_processor.py          # Frame extraction, metadata, video I/O
├── generate_defects.py         # Synthetic defect injection for validation
├── services/
│   ├── qwen_vlm_service.py     # Primary VLM (Qwen3-VL-2B) integration
│   └── vlm_service.py          # Offline VLM (Moondream2) integration
```

## Installation

```bash
git clone <repo-url>
cd Smart-and-Intelligent-Video-Analysis-System-using-Agentic-AI
pip install -r requirements.txt
```

You'll also need Real-ESRGAN weights (`RealESRGAN_x4plus.pth`) placed at the path configured in `config.py`, and a CUDA-capable GPU for practical inference speed.

## Usage

```bash
streamlit run app.py
```

Upload a video, adjust the frame sampling interval if needed, and run the quality analysis pipeline. Frames flagged as defective are automatically enhanced and displayed alongside the reasoning that triggered the enhancement.

## Roadmap

- [ ] Temporal anomaly detection — flagging unusual events/behavior across frames over time, extending the pipeline from per-frame quality analysis toward broader autonomous surveillance monitoring.

## Disclaimer

This is a research/portfolio project built to explore agentic AI pipeline design with VLMs. It is not a production-grade surveillance system.
