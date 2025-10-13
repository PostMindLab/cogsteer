import torch
import os
from transformers import LlamaForSequenceClassification, AutoTokenizer, PretrainedConfig, AutoConfig
from datasets import load_dataset
import evaluate 
from tqdm import tqdm
import json
import sys
import pandas as pd
from pathlib import Path
import adapters
from adapters import AutoAdapterModel

model_name = "Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device="cuda"

def eval(i):
    model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) 
    model.config.pad_token_id = tokenizer.pad_token_id

    adapters.init(model)
    model.load_adapter(f"llama/weights_pt/pt_layer{i}_{task}")
    model.set_active_adapters(f"pt_layer{i}_{task}")
    model.to(device)

    
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    validation_key = "validation_mismatched" if task == "mnli" else "validation"
    test_key = "test_mismatched" if task == "mnli" else "test"

    if is_test:
        eval_dataset = encoded_dataset[test_key]
    else:
        eval_dataset = encoded_dataset[validation_key]
    task_length[task] = len(eval_dataset)
    predictions = []
    labels = []

    for batch in tqdm(eval_dataset):
        inputs = {k: torch.tensor(batch[k]).unsqueeze(0).to(device) for k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(torch.argmax(logits, dim=-1).item())
            labels.append(batch['labels'])

    if is_test:
        df = pd.DataFrame(predictions, columns=["prediction"])
        df.index.name = "index"
        df.to_csv(output_path+f"{task.upper()}_mm.tsv", sep="\t", index=True)
    else:
        results = metric.compute(predictions=predictions, references=labels)
        val_result.append({"task": task, "layer": i, "type": "pt", "result": results})
    del model
    torch.cuda.empty_cache()

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
task_length={}
for task in task_to_keys:
    dataset = load_dataset("nyu-mll/glue", task)
    metric = evaluate.load("glue", task )
    is_regression = task == "stsb"
    if not is_regression:
        label_list = dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    # print(dataset)
    # print(dataset["train"][0])

    sentence1_key, sentence2_key = task_to_keys[task]
    def preprocess_function(examples):
    # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        
        encoding = tokenizer(
            *texts,
            padding="max_length",
            return_overflowing_tokens=False,
            truncation=True,
            max_length=128,
            return_tensors=None
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": examples["label"]
        }


    non_label_column_names = dataset["train"].column_names
    print(non_label_column_names)
    encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=non_label_column_names,load_from_cache_file=True)

    val_result=[]
    is_test = False # test or val
    layers = ["full"] # layer index
    for i in layers:
        print(f"Task: {task}, Layer: {i}")
        if is_test:
            output_path = f"./llama/test/layer_{i}/"
            Path(output_path).mkdir(parents=True, exist_ok=True)
        eval(i)
    if not is_test:
        json.dump(val_result, open(f"./llama/val/{task}.json", "w"), indent=2)
print(task_length)