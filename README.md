# CogSteer: Cognition-Inspired Selective Layer Intervention for LLMs

Official implementation of [CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](https://arxiv.org/abs/2410.17714)

## 💻 Environment Setup

```bash
# Create and activate conda environment
conda create -n cogsteer -y python=3.10
conda activate cogsteer

# Install dependencies
pip install -r requirements.txt
```
Download the llama2 original checkpoint in [meta](https://www.llama.com/llama-downloads/?utm_source=llama-home-hero&utm_medium=llama-referral&utm_campaign=llama-utm&utm_offering=llama-downloads&utm_product=llama)

## 📖 Usage

### 🔍 Correlation Analysis

Analyze the correlation between model outputs and eye-tracking data:

```bash
cd correlation
# Update model path in correlation.py for LLaMA-2 checkpoints
python correlation.py
```

**Configuration**:
- Set your LLaMA-2 model path in `correlation.py`
- Ensure eye-tracking data is available in the expected format

### 📊 GLUE Benchmark

Train and evaluate models on GLUE tasks using Layer Intervention:

```bash
cd glue
export CUDA_VISIBLE_DEVICES=xxx
# For LLaMA
python llama_train_pt.py  # Training
python llama_eval_pt.py   # Evaluation

# For Mistral  
python mistral_train_pt.py
python mistral_eval_pt.py

# For GPT-2
python gpt2_train_pt.py
python gpt2_eval_pt.py
```

**Configuration**:
- Modify the `layer` parameter to target specific layers
- Set `layer="full"` for full model intervention

### 🛡️ Toxicity Control

#### Framework

Our implementation uses different adapter frameworks optimized for each model architecture:

- **GPT-2 & Mistral**: Built on the [Adapters](https://github.com/adapter-hub/adapters) framework using HuggingFace model checkpoints. This provides efficient parameter-efficient fine-tuning with minimal memory overhead.

- **LLaMA**: Uses Meta's original model checkpoints with the [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) framework, which is specifically designed for LLaMA models and provides better compatibility with the original architecture.

#### Training Toxic Adapters

**For GPT-2 and Mistral**:
```bash
cd tox/gpt2  # or tox/mistral
python train_tox.py
```

**For LLaMA with LLaMA-Adapter**:
Update model checkpoint path in `TARGET_FOLDER` in finetuning.sh
```bash
cd tox/llama_adapter/finetune
bash finetuning.sh
```


#### Running Detoxification

**For GPT-2 and Mistral**:
```bash
cd tox/mistral  # or tox/gpt2
python detox.py
```

**For LLaMA**:
Set `ckpt_dir` with model checkpoint in `llama_generate_detox.py`
```bash
cd tox/llama_adapter/generate
python llama_generate_detox.py
```

#### Evaluating Toxicity
We are using [Perplexity](https://perspectiveapi.com/) to evaluate the toxicity of the sentence. First you need to obtain your API_KEY from Perplexity.
1. Set `API_KEY` and specify your evaluating folder `answers_dir` in `get_score.py`. Run `get_score.py`
2. Set `answers_dir` and `output_dir` before running `cal_metrics.py`

## 📄 Citation

If you find our work useful, please consider starring the repository and citing our paper:

```bibtex
@article{wang2024cogsteer,
  title={Cogsteer: Cognition-inspired selective layer intervention for efficiently steering large language models},
  author={Wang, Xintong and Pan, Jingheng and Ding, Liang and Wang, Longyue and Jiang, Longqin and Li, Xingshan and Biemann, Chris},
  journal={arXiv preprint arXiv:2410.17714},
  year={2024}
}
```



