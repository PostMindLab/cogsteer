from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import AdapterConfigBase, ModelAdaptersConfig
# from adapters import MistralAdapterModel
from adapters import AdapterConfig, BnConfig
from adapters import AdapterTrainer
import sys
import wandb
import adapters
import os

data_dir = '../data/'
mistral_model = "mistralai/Mistral-7B-v0.3"

dataset = load_dataset('csv', data_files={'train': data_dir + 'train.csv', 'test':data_dir + 'test.csv'})

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  encoding = tokenizer(batch["comment_text"])
  # For language modeling the labels need to be the input_ids
  #encoding["labels"] = encoding["input_ids"]
  return encoding

tokenizer = AutoTokenizer.from_pretrained(mistral_model)
# The GPT-2 tokenizer does not have a padding token. In order to process the data 
# in batches we set one here 
tokenizer.pad_token = tokenizer.eos_token
column_names = dataset["train"].column_names
dataset = dataset.map(encode_batch, remove_columns=column_names, batched=True)

print('group_texts')

block_size = 50
# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
  # Concatenate all texts.
  concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
  # customize this part to your needs.
  total_length = (total_length // block_size) * block_size
  # Split by chunks of max_len.
  result = {
    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    for k, t in concatenated_examples.items()
  }
  result["labels"] = result["input_ids"].copy()
  return result

dataset = dataset.map(group_texts,batched=True,)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
num_layers = 32


model_name = "Mistral-7B"
def train_model(dataset, layer_index):
  if layer_index == "full":
        leave_out = []
  else:
      leave_out = [l for l in range(num_layers)]
      leave_out.remove(layer_index)

  print('leave_out:', leave_out)
  config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16,leave_out=leave_out, non_linearity="relu")

  model = AutoModelForCausalLM.from_pretrained(mistral_model)
  adapters.init(model)
  # add new adapter
  model.add_adapter("toxic_{}_{}".format(model_name, layer_index),config=config)
  model.train_adapter("toxic_{}_{}".format(model_name, layer_index))

  training_args = TrainingArguments(
    output_dir="./temp/toxic_{}_{}".format(model_name, layer_index), 
    do_train=True,
    learning_rate=5e-5,
    num_train_epochs=1,
    overwrite_output_dir=True,
    eval_strategy="steps",
    eval_steps=300,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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

  trainer = AdapterTrainer(
          model=model,
          args=training_args,
          tokenizer=tokenizer,
          train_dataset=dataset["train"],
          eval_dataset=dataset["test"], 
      )

  print('training {} {} layer...'.format(model_name, layer_index))
  trainer.train()
  print('evaluate {} {} layer...'.format(model_name, layer_index))
  trainer.evaluate()

  print('saving {}...'.format(layer_index))
  os.makedirs(f"weights/toxic_{model_name}_layer_{layer_index}", exist_ok=True)
  model.save_adapter("weights/toxic_{}_layer_{}".format(model_name, layer_index), "toxic_{}_{}".format(model_name, layer_index))

# Train all layers
for i in range(num_layers):
   train_model(dataset, i)
train_model(dataset, "full")
