import torch
import os
import wandb

from transformers import logging
logging.set_verbosity_warning()

product = "Falcon3-3B-qlora-1k"
max_seq_length = 1024*4
dtype = torch.bfloat16
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
os.environ["WANDB_WATCH"] = "false"  # Disable gradient logging

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "tiiuae/Falcon3-3B-Base",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)
print("Model loaded successfully.")

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    lora_alpha = 64,
    target_modules = [
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
        "lm_head", "embed_tokens",
    ],
    use_gradient_checkpointing="unsloth",
)
print("Model patched successfully.")

from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_sharegpt
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    # map_eos_token=False,
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False, truncation = True, padding=True) for convo in convos]
    return { "text" : texts }
pass

from datasets import load_dataset

# Get LAION dataset
# url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
# dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

#dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:100000]")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)
split = dataset.train_test_split(test_size = 0.04)
print(f"Dataset split: {len(split['train'])} train, {len(split['test'])} test samples.")

print("Tokenizer configuration:")
print(f"Pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

from unsloth import UnslothTrainer, UnslothTrainingArguments
args = UnslothTrainingArguments(
    bf16 = True,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 1,
    packing=True,
    # eval_packing = False,
    num_of_sequences = max_seq_length,
    max_seq_length = max_seq_length,
    warmup_steps = 20,
    max_steps = 200,
    num_train_epochs = 1,
    lr_scheduler_type = "constant",
    learning_rate = 5e-5,
    embedding_learning_rate = 5e-6, # 2-10x smaller than learning_rate
    logging_steps = 1,
    optim = "paged_ademamix_8bit",
    output_dir = "outputs",
    report_to = "wandb",
    eval_strategy="epoch",
)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = split.get("train"),
    eval_dataset = split.get("test"),
    args = args
)
print("Trainer created successfully.")

# Train on responses only
# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = "<|im_start|>user\n",
#     response_part = "<|im_start|>assistant\n",
# )

# print("Masked response tokens:")
# red = "\033[31m"
# green = "\033[32m"
# blue = "\033[34m"
# reset = "\033[0m"
# colored_tokens = []
# print("tokens", len(trainer.train_dataset[1]["labels"]))
# print("tokens size", len(trainer.train_dataset[1]["labels"]))
# special_ids = tokenizer.added_tokens_decoder.keys()

# for token_id in trainer.train_dataset[1]["labels"]:
#     if token_id == -100:
#         colored_tokens.append(f"{red}(-){reset}")
#     elif token_id in special_ids:
#         token_str = tokenizer.decode([token_id])
#         colored_tokens.append(f"{blue}{token_str}({token_id}){reset}")
#     else:
#         token_str = tokenizer.decode([token_id])
#         colored_tokens.append(f"{green}{token_str}({token_id}){reset}")
# print("".join(colored_tokens))

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project=product, entity="pink-marker")
trainer_stats = trainer.train()
wandb.finish()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Write about the 5 pillars of prosperity."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
tokenizer.batch_decode(outputs)

model.save_pretrained(product)
tokenizer.save_pretrained(product)
print(f"Model saved to output/{product}.")
