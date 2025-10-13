from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments
from adapters import AdapterTrainer, BnConfig
import adapters
import torch
# Configuration
import os
import sys

data_dir = '../data/'
model_name = "gpt2-large"
block_size = 50
num_train_epochs = 5
training_lr = 5e-4
num_layers = 12  # Assuming gpt2-large has 36 layers
adapter_reduction_factor = 16
adapter_non_linearity = "relu"

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load Dataset
def load_and_prepare_dataset(data_dir):
    dataset = load_dataset('csv', data_files={'train': data_dir + 'train.csv', 'test': data_dir + 'test.csv'})
    column_names = dataset["train"].column_names
    dataset = dataset.map(encode_batch, remove_columns=column_names, batched=True)
    dataset = dataset.map(group_texts, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

# Encode Batch
def encode_batch(batch):
    encoding = tokenizer(batch["comment_text"])
    return encoding

# Group Texts
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
    result["labels"] = result["input_ids"].copy()
    return result

dataset = load_and_prepare_dataset(data_dir)

# Train Model
def train_model(model_addr, dataset, layer_index):
    if layer_index == "full":
        leave_out = []
    else:
        leave_out = [l for l in range(num_layers)]
        leave_out.remove(layer_index)
    print(leave_out)

    config = BnConfig(
        mh_adapter=True,
        output_adapter=True,
        reduction_factor=adapter_reduction_factor,
        leave_out=leave_out,
        non_linearity=adapter_non_linearity,
    )
    model = AutoModelForCausalLM.from_pretrained(model_addr)
    adapters.init(model)
    adapter_name = f"toxic_{model_name}_{layer_index}"
    model.add_adapter(adapter_name, config=config)
    model.train_adapter(adapter_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    training_args = TrainingArguments(
        output_dir="./temp/toxic_{}_{}".format(model_name, layer_index), 
        do_train=True,
        learning_rate=5e-5,
        num_train_epochs=5,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=300,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        logging_steps=100,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        report_to="none",
        save_steps=900,

        weight_decay=0.01,                     
        load_best_model_at_end=True,           
        metric_for_best_model="eval_loss",     
        greater_is_better=False,               
        lr_scheduler_type="cosine",            
    )

    trainer = AdapterTrainer(model=model, args=training_args, tokenizer=tokenizer, train_dataset=dataset["train"], eval_dataset=dataset["test"])
    trainer.train()
    os.makedirs(f"weights/toxic_{model_name}_layer_{layer_index}", exist_ok=True)
    model.save_adapter(f"weights/toxic_{model_name}_layer_{layer_index}", adapter_name)

# Train all layers
for i in range(num_layers):
   train_model(model_name, dataset, i)
train_model(model_name, dataset, "full")


