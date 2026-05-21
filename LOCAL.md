# Local Model Generation

This guide explains how to generate TritonBench-T predictions with a model
running on your Mac, then evaluate those predictions on Modal.

The important split is:

- `local_generate.py` runs on your laptop and talks to your local model server.
- `modal_app.py::evaluate_only` uploads the generated JSONL file and evaluates
  it on a Modal GPU.

Do not make Modal call `localhost`. Inside Modal, `localhost` means the Modal
container, not your laptop.

## 1. Install and Start a Local Model Server

The easiest path is Ollama because it runs well on Apple Silicon and exposes an OpenAI-compatible API at:

```bash
http://localhost:11434/v1
```

Install Ollama from the official installer, then pull a coding model:

```bash
ollama pull qwen2.5-coder:14b
```

For your 48 GB M4 Pro laptop, start with `qwen2.5-coder:14b`. If that is stable
and you want to try a stronger but slower model, pull:

```bash
ollama pull qwen2.5-coder:32b
```

## 2. Confirm the Local API Works

Run this from the repository root:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder:14b",
    "messages": [
      {"role": "user", "content": "Write a Python function that adds two numbers."}
    ]
  }'
```

Expected output:

- A JSON response.
- A top-level `choices` array.
- Generated text under `choices[0].message.content`.

If you get a connection error, open Ollama and try again.

## 3. Clone TritonBench Locally

`modal_app.py` clones TritonBench inside the Modal container, but local
generation needs the dataset files on your laptop too.

From this repository:

```bash
git clone https://github.com/thunlp/TritonBench.git
```

The generator defaults to reading:

```bash
./TritonBench/data/TritonBench_T_simp_alpac_v1.json
```

If you clone TritonBench somewhere else, pass `--repo`:

```bash
python local_generate.py --repo /path/to/TritonBench
```

Or set:

```bash
export TRITONBENCH_LOCAL_REPO=/path/to/TritonBench
```

## 4. Generate a Small Smoke Test

Run only 5 examples first:

```bash
python local_generate.py \
  --model qwen2.5-coder:14b \
  --limit 5 \
  --output local_predictions.jsonl
```

Expected terminal output:

```text
dataset: simp
items: 5
already complete: 0
pending: 5
model: qwen2.5-coder:14b
endpoint: http://localhost:11434/v1/chat/completions
output: /absolute/path/to/local_predictions.jsonl
1/5 complete (... min elapsed)
2/5 complete (... min elapsed)
...
wrote /absolute/path/to/local_predictions.jsonl
```

Expected file output:

```bash
local_predictions.jsonl
```

Each line should be one JSON object with:

```json
{"instruction": "...", "predict": "import torch\nimport triton\n..."}
```

The `instruction` must remain exactly from TritonBench. The evaluator uses it
to match generated code to the correct reference operator.

## 5. Evaluate the Smoke Test on Modal

Use the existing Modal app:

```bash
modal run modal_app.py::evaluate_only --predictions ./local_predictions.jsonl
```

Expected output:

- The file is uploaded into the Modal Volume.
- Phase 1 prints call accuracy progress.
- Phase 2 prints execution accuracy progress.
- Phase 3 runs only for operators that passed phases 1 and 2.
- A final JSON summary is printed.

Example shape:

```json
{
  "total_predictions": 5,
  "phase1_call_acc": {
    "passed": 2,
    "rate": 40.0
  },
  "phase2_exec_acc": {
    "passed": 1,
    "rate": 20.0
  },
  "phase3_efficiency": {
    "speedup_vs_pytorch": 0.83,
    "raw_output_tail": "..."
  },
  "artifacts_volume": "tritonbench-t-data",
  "artifacts_subdir": "results"
}
```

Your actual numbers will vary by model and generation quality.

## 6. Run the Full Dataset

Once the smoke test works:

```bash
python local_generate.py \
  --model qwen2.5-coder:14b \
  --output local_predictions.jsonl
```

Then evaluate:

```bash
modal run modal_app.py::evaluate_only --predictions ./local_predictions.jsonl
```

The default dataset is `simp`. To generate the complex prompt variant:

```bash
python local_generate.py \
  --model qwen2.5-coder:14b \
  --dataset comp \
  --output local_predictions_comp.jsonl
```

Then evaluate that file:

```bash
modal run modal_app.py::evaluate_only --predictions ./local_predictions_comp.jsonl
```

## 7. Resume an Interrupted Run

`local_generate.py` writes the JSONL file after every completed item.

If your laptop sleeps, the model server stops, or the process is interrupted,
run the same command again:

```bash
python local_generate.py \
  --model qwen2.5-coder:14b \
  --output local_predictions.jsonl
```

Expected output will show completed rows as already done:

```text
items: 166
already complete: 47
pending: 119
```

To discard the existing output and regenerate from scratch:

```bash
python local_generate.py \
  --model qwen2.5-coder:14b \
  --output local_predictions.jsonl \
  --overwrite
```

## 8. Useful Options

Use a different local endpoint:

```bash
python local_generate.py \
  --base-url http://localhost:1234/v1 \
  --model your-model-name
```

Use a different output file:

```bash
python local_generate.py --output predictions/qwen_local_simp.jsonl
```

Increase timeout for slow models:

```bash
python local_generate.py --timeout 1800
```

Keep concurrency at `1` for most laptop runs:

```bash
python local_generate.py --concurrency 1
```

Higher concurrency usually does not help local models. It often increases memory
pressure and makes each request slower.

## 9. Failure Rows

If a model request fails, the generated row will contain:

```python
# generation failed: ...
```

The script exits with a non-zero status if any rows failed. You can inspect the
file and rerun after fixing the local model server. By default, reruns resume
from existing rows, so use `--overwrite` if you want to regenerate failed rows
from scratch.

## 10. Recommended Workflow

Use this sequence:

```bash
ollama pull qwen2.5-coder:14b
git clone https://github.com/thunlp/TritonBench.git

python local_generate.py --model qwen2.5-coder:14b --limit 5
modal run modal_app.py::evaluate_only --predictions ./local_predictions.jsonl

python local_generate.py --model qwen2.5-coder:14b --output local_predictions.jsonl
modal run modal_app.py::evaluate_only --predictions ./local_predictions.jsonl
```

