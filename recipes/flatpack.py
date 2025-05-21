from regex import T
from unsloth import FastLanguageModel
import torch
import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset

from trl import SFTConfig
import gc
import wandb
import torch

from utils.flatpack_utils import (
    DataCollatorForFlatpack,
    FlatpackTrainer,
    group_batches_for_flatpack,
    vram_report_end,
    vram_report_start,
)

project = "flatpack"
max_seq_length = 1024 * 8

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    attn_implementation="flash_attention_2",
)
print(
    f"Model loaded. dtype = {model.dtype}, device = {model.device}, attn_implementation = {model.config._attn_implementation}"
)
print(
    f"Tokenizer loaded. vocab size = {tokenizer.vocab_size}, model max length = {tokenizer.model_max_length}"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
)

airoboros = load_dataset("jondurbin/airoboros-3.2", split="train")
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

chatml_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>' }}"
    "{% endif %}"
)
chatml_eos_token = "<|im_end|>"

tokenizer = get_chat_template(
    tokenizer, chat_template=(chatml_template, chatml_eos_token, True)
)

airoboros = standardize_sharegpt(airoboros)
airoboros = airoboros.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"],
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    },
    remove_columns=airoboros.column_names,
    batched=True,
    desc="Formatting airoboros dataset",
)

split = airoboros.train_test_split(test_size=0.02)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} test samples.")

# group the samples into batches
train_dataset, per_device_train_batch_size = group_batches_for_flatpack(
    train_dataset,
    max_seq_length=max_seq_length,
)

train_dataset, per_device_eval_batch_size = group_batches_for_flatpack(
    eval_dataset,
    max_seq_length=max_seq_length,
)

args = SFTConfig(
    output_dir="outputs/" + project,
    report_to="wandb",
    num_train_epochs=2,
    learning_rate=1e-5,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=per_device_eval_batch_size,
    eval_accumulation_steps=1,
    batch_eval_metrics=True,
    optim="adamw_8bit",
    lr_scheduler_type="constant_with_warmup",
    # max_grad_norm=0.5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=1,
    eval_strategy="steps",
    do_eval=True,
    eval_steps=100,
    save_total_limit=3,
)

trainer = FlatpackTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    data_collator=DataCollatorForFlatpack(),
)

wandb.init(project="flatpack", entity="pink-marker", save_code=True)

gc.collect()
torch.cuda.empty_cache()

vram_report_start("start")
trainer_stats = trainer.train(resume_from_checkpoint=False)
vram_report_end(trainer_stats)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a rascal ai assistant."},
    {"role": "user", "content": "Please write about the 5 pillars of prosperity."},
]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

encoded = tokenizer(formatted, return_tensors="pt").to("cuda")

outputs = model.generate(
    **encoded,
    max_new_tokens=1024 * 2,
    use_cache=True,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    repetition_penalty=1.1,
    do_sample=True,
)
outputs = tokenizer.batch_decode(outputs)
for output in outputs:
    clean = output.replace("\\n", "\n").replace("\\t", "\t")
    print(clean)
