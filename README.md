# Detecting Clinical Hallucinations in LVLMs via Counterfactual Visual Grounding Uncertainty (MICCAI 2026)

## 🔧 Dependencies and Installation
- Python == 3.10.18
- [PyTorch == 2.6.0+cu124](https://pytorch.org/)
- [transformers >= 4.57.0](https://huggingface.co/docs/transformers)

### Installation
1. Clone repo

```bash
git clone https://github.com/NJU-MedAI-Lab/Mobius.git
cd Mobius
```
2. Install dependent packages (use conda)

```bash
conda create -n counterdetect python=3.10.18 -y
conda activate counterdetect
pip install -r requirements.txt
```

`flash-attn` is hardware/CUDA sensitive. If the wheel build fails, install the wheel that matches your CUDA and PyTorch versions, then rerun the dependency installation.

## 🗂️ Datasets

Our datasets are built based on [IMed-361M](https://huggingface.co/datasets/General-Medical-AI/IMed-361M) dataset. The annotation files used by this project are provided in the `datasets/` directory, while the image files should be downloaded separately from IMed-361M and pointed to by `IMAGE_ROOT_DIR`.

- `datasets/VFM_train.json`: the main visual grounding fine-tuning set with supporting and contradictory conversations.
- `datasets/VFM_train_wronglesion.json`: an additional training split with wrong-lesion counterfactual samples.
- `datasets/VFM_train_unvisible.json`: an additional training split with unvisible/absent finding samples.
- `datasets/VFM_eval.json`: the validation split used during training.
- `datasets/HalluEsti_test.json`: the hallucination estimation test split with reports, model source labels, hallucination labels, and counterfactual grounding pairs.

The dataset keys registered in `src/data/__init__.py` are:

```text
hallu_esti_training
hallu_esti_wrong_lesion
hallu_esti_training_unvisible
hallu_esti_eval
hallu_esti_test
```

- You can download the images from [Huggingface](https://huggingface.co/datasets/General-Medical-AI/IMed-361M).

## 🖥️ Environment Variables

Set the repository root and dataset/model paths before running scripts:

```bash
export PROJECT_ROOT=/path/to/Counterfactual_Hallu_Detect
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

export IMAGE_ROOT_DIR=/path/to/IMed-361M/images
export ENTITY_CANDIDATE_FILE=/path/to/all_classes_new.json
export MODALITY_MAPPING_FILE=/path/to/modality_mapping.json
```

Entity extraction and counterfactual generation use an OpenAI-compatible API. All previous hard-coded values have been replaced by placeholders. Configure your own endpoint when these functions are enabled:

```bash
export LLM_BASE_URL=YOUR_LLM_BASE_URL
export LLM_API_KEY=YOUR_LLM_API_KEY

# Optional per-provider overrides
export OPENAI_BASE_URL=YOUR_OPENAI_COMPATIBLE_BASE_URL
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export GEMINI_BASE_URL=YOUR_GEMINI_COMPATIBLE_BASE_URL
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
export GROK_BASE_URL=YOUR_GROK_COMPATIBLE_BASE_URL
export GROK_API_KEY=YOUR_GROK_API_KEY
export CLAUDE_BASE_URL=YOUR_CLAUDE_COMPATIBLE_BASE_URL
export CLAUDE_API_KEY=YOUR_CLAUDE_API_KEY
```

## ⚙️ Train

Use the provided LoRA training script:

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-2B-Instruct
export OUTPUT_DIR=${PROJECT_ROOT}/outputs/checkpoints/qwen3vl-2b-lora
bash scripts/lora_qwen3_2b.sh
```

The script launches `src/train/train_qwen.py` with `torchrun`. Adjust GPU IDs, batch size, learning rate, dataset keys, and LoRA settings in `scripts/lora_qwen3_2b.sh`.

You can also run the entry directly:

```bash
torchrun --nproc_per_node=2 src/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
  --dataset_use hallu_esti_training,hallu_esti_wrong_lesion \
  --eval_dataset_use hallu_esti_eval \
  --output_dir outputs/checkpoints/qwen3vl-2b-lora \
  --bf16 \
  --lora_enable True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8
```

## 🏰 Model Zoo

We provide the visual grounding verifier checkpoints on [Huggingface](https://huggingface.co/Agentic-CliniAI/Counterfactual-Hallucination-Detect)

## ⚡️ Test

For the full hallucination test pipeline with optional entity extraction and counterfactual generation:

```bash
export MODEL_NAME_OR_PATH=/path/to/checkpoint/hf
export OUTPUT_DIR=${PROJECT_ROOT}/outputs/results/qwen3vl-test
bash src/eval/scripts/test.sh
```

Or run the test entry directly:

```bash
torchrun --nproc_per_node=1 src/eval/test.py \
  --test_only True \
  --model_name_or_path /path/to/checkpoint/hf \
  --eval_dataset_use hallu_esti_test \
  --output_dir outputs/results/qwen3vl-test \
  --image_root_dir ${IMAGE_ROOT_DIR} \
  --entity_candidate_file ${ENTITY_CANDIDATE_FILE} \
  --uncertainty_type all \
  --do_entity_extract True \
  --do_counterfactual_test True \
  --entity_extract_model gpt-4.1-mini \
  --cf_model gpt-4.1-mini \
  --bf16
```

Set `--do_entity_extract False` and `--do_counterfactual_test False` to evaluate with existing dataset phrases only.

## 📈 Batch Evaluation and Metrics

Batch grounding evaluation:

```bash
bash src/eval/scripts/eval_batch.sh
```

Counterfactual batch grounding evaluation:

```bash
bash src/eval/scripts/eval_batch_counter.sh
```

Calculate hallucination metrics from combined results:

```bash
python src/eval/cal_hallu_metric_from_results.py \
  --result_type combined \
  --sup_root_dir outputs/results/qwen3vl-test/results.json \
  --image_root_dir ${IMAGE_ROOT_DIR} \
  --output_dir outputs/results/qwen3vl-test \
  --uncertainty_type logits \
  --tau 0.6 \
  --use_counterfactual
```

Calculate mIoU-style localization metrics:

```bash
python src/eval/cal_metric.py \
  --result_root outputs/batch_eval/qwen3vl-test \
  --image_root ${IMAGE_ROOT_DIR} \
  --uncertainty_type all
```
