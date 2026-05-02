# MuMA-ToM Reproduction

This workspace contains pragmatic MuMA-ToM reproduction paths built around:

- `gpt-4o` with frame sampling
- `gemini-2.5-pro` with direct video input as a replacement direct-QA variant
- `gemini-2.5-pro` visual extraction + `gpt-4o` LIMP reasoning
- `LLaVA 1.6` image-frame baselines

Scope:
- download the public MuMA-ToM benchmark from Hugging Face
- load corrected `questions.json`, `texts.json`, and the benchmark videos
- sample non-video-native model inputs every 20 frames to match the paper's stated LLaVA/GPT-4o preprocessing
- evaluate a GPT-4o multiple-choice baseline and a Gemini direct-video replacement baseline
- write jsonl outputs

Notes:
- The Gemini baseline here is a direct `video + texts.json + MCQ` replacement variant, not the paper's original Gemini 1.5 Pro web setup.
- The paper reports GPT-4o as a baseline and uses frame inputs for non-video-native models rather than raw video.
- LLaVA defaults are set to paper-near every-20-frames sampling, with adaptive frame downsampling only when the raw prompt would exceed the model context limit on this server.
- Treat this as a near-reproduction path, not an exact paper rerun.

## Setup

The `.env` file should define:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4o`)
- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-2.5-pro`)
- `HF_TOKEN`

Optional:

- `FRAME_STRIDE` (default: `20`)
- `MAX_FRAMES` (default: `24`)

Install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

## Download the benchmark

```bash
.venv/bin/python run_muma_gpt4o.py --download-only
.venv/bin/python run_muma_gemini.py --download-only
```

## Run a GPT-4o sanity slice

```bash
.venv/bin/python run_muma_gpt4o.py --limit 5 --frame-stride 60 --max-frames 4
```

## Run a Gemini sanity slice

```bash
.venv/bin/python run_muma_gemini.py --limit 1
```

## Extract Gemini actions for LIMP

```bash
.venv/bin/python extract_muma_actions.py --limit 10
```

## Run a LIMP sanity slice

```bash
.venv/bin/python run_muma_limp.py --limit 4
```

## Run LLaVA baselines

```bash
.venv/bin/python run_muma_llava.py --variant 13b --limit 10
.venv/bin/python run_muma_llava.py --variant 34b --limit 10
```

## Aggregate results

```bash
.venv/bin/python aggregate_muma_results.py
```

## Run the full benchmark

```bash
.venv/bin/python run_muma_gpt4o.py --limit 900
.venv/bin/python run_muma_gemini.py --limit 900
```

## Outputs

- `outputs/muma_gpt4o_predictions.jsonl`
- `outputs/muma_gpt4o_predictions.summary.json`
- `outputs/muma_gemini_predictions.jsonl`
- `outputs/muma_gemini_predictions.summary.json`
- `outputs/actions_extracted_gemini.json`
- `outputs/muma_limp_predictions.jsonl`
- `outputs/muma_limp_predictions.summary.json`
- `outputs/muma_llava_13b.jsonl`
- `outputs/muma_llava_13b.summary.json`
- `outputs/muma_llava_34b.jsonl`
- `outputs/muma_llava_34b.summary.json`
- `outputs/muma_results_table.md`
