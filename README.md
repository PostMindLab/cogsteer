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

For GPT-2 and Mistral:
```bash
cd tox/gpt2  # or tox/mistral
python train_tox.py
```

For LLaMA with LLaMA-Adapter:
Update model checkpoint path in `TARGET_FOLDER` in finetuning.sh
```bash
cd tox/llama_adapter/finetune
bash finetuning.sh
```


#### Running Detoxification

For GPT-2 and Mistral:
```bash
cd tox/mistral  # or tox/gpt2
python detox.py
```

For LLaMA: Set `ckpt_dir` with model checkpoint in `llama_generate_detox.py`
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
@inproceedings{wang-etal-2025-cogsteer,
    title = "{C}og{S}teer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models",
    author = "Wang, Xintong  and
      Pan, Jingheng  and
      Ding, Liang  and
      Wang, Longyue  and
      Jiang, Longqin  and
      Li, Xingshan  and
      Biemann, Chris",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1308/",
    doi = "10.18653/v1/2025.findings-acl.1308",
    pages = "25507--25522",
    ISBN = "979-8-89176-256-5",
    abstract = "Large Language Models (LLMs) achieve remarkable performance through pretraining on extensive data. This enables efficient adaptation to diverse downstream tasks. However, the lack of interpretability in their underlying mechanisms limits the ability to effectively steer LLMs for specific applications. In this work, we investigate the intrinsic mechanisms of LLMs from a cognitive perspective using eye movement measures. Specifically, we analyze the layer-wise correlation between human cognitive indicators and LLM representations. Building on these insights, we propose a heuristic approach for selecting the optimal steering layer to modulate LLM semantics. To this end, we introduce an efficient selective layer intervention based on prominent parameter-efficient fine-tuning methods, which conventionally adjust either all layers or only the final layer. Additionally, we present an implicit layer contrastive intervention during inference to steer LLMs away from toxic outputs. Extensive experiments on natural language understanding, reasoning, and generation tasks, conducted on GPT-2, LLaMa2-7B, and Mixtral-7B, demonstrate the effectiveness and efficiency of our approach. As a model-agnostic framework, it enhances the interpretability of LLMs while improving efficiency for safe deployment."
}
```



