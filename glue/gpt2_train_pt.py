# import torch
import os

from datasets import load_dataset
# import evaluate 
from tqdm import tqdm
from transformers import GPT2Tokenizer, TrainingArguments, GPT2ForSequenceClassification
# from transformers.adapters import AdapterConfig, PrefixTuningConfig
from adapters import AdapterTrainer, AutoAdapterModel, PrefixTuningConfig
import adapters
import numpy as np
import wandb
import json
import sys
import torch
import evaluate

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def train(i):
    if i == "full":
        leave_out = []
    else:
        leave_out = [l for l in range(num_layers)]
        leave_out.remove(i)

    print('leave_out:', leave_out)

    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) 
    model.config.pad_token_id = tokenizer.pad_token_id
    adapters.init(model)

    adap_name = f"pt_layer{i}_{task}"
    config = PrefixTuningConfig(flat=False, prefix_length=30, leave_out=leave_out)
    model.add_adapter(adap_name, config=config)
    model.train_adapter(adap_name)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    training_args = TrainingArguments(
        output_dir=f"./gpt2-large/temp/{adap_name}", 
        do_train=True,
        learning_rate=2e-5,
        num_train_epochs=2,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,

        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=100,
        remove_unused_columns=False,
        report_to="wandb",

        weight_decay=0.0001,                    
        load_best_model_at_end=True,            
        greater_is_better=True,               
        metric_for_best_model=metric_name,      
        lr_scheduler_type="cosine",             
        warmup_ratio=0.1,
        seed=42,
    )
    validation_key = "validation_matched" if task == "mnli" else "validation"
    trainer = AdapterTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key], 
            compute_metrics=compute_metrics
        )

    print('training {} {} layer...'.format(model_name, i))
    trainer.train()
    print('evaluate {} {} layer...'.format(model_name, i))
    results = trainer.evaluate()
    val_result.append({"task": task, "layer": i, "type": "pt", "result": results})

    print('saving {}...'.format(i))
    model.save_adapter(f"gpt2-large/weights_pt/{adap_name}", f"{adap_name}", with_head=True)

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
last_label = 0
num_layers = 36
val_result = []
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
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

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
    encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=non_label_column_names)

    layer = "full" # layer index
    train(layer)

    json.dump(val_result, open(f"./gpt2-large/val/{task}.json", "w"))
