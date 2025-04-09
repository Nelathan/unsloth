from unsloth import FastModel
import torch

product = "Gemma-3-4B"

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=1024 * 6,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=False,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import standardize_data_formats

sugarquill = load_dataset("Nelathan/synthetic-sugar-quill", split="train")
sugarquill = sugarquill.map(
    lambda batch: {
        "conversations": [
            [
                {"role": "user", "content": "Define yourself."},
                {"role": "assistant", "content": profile},
                {"role": "user", "content": "Write a story."},
                {"role": "assistant", "content": text},
            ]
            for profile, text in zip(batch["profile"], batch["text"])
        ]
    },
    remove_columns=[col for col in sugarquill.column_names if col != "id"],
    batched=True,
    desc="Formatting Sugarquill",
)
sugarquill = standardize_data_formats(sugarquill)


def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}


pass

ds_split = sugarquill.map(
    apply_chat_template,
    batched=True,
    remove_columns=[col for col in sugarquill.column_names if col != "id"],
)


# def sort_and_shuffle(dataset, target_id):
#     target_row = dataset.filter(lambda x: x["id"] == target_id)
#     assert len(target_row) == 1, f"Target ID {target_id} not found in dataset."
#     rest = dataset.filter(lambda x: x["id"] != target_id)
#     rest = rest.shuffle(seed=42)
#     return concatenate_datasets([target_row, rest])


# ds_split = sort_and_shuffle(ds_split, target_id=7575)

# ds_split = ds_split.train_test_split(100, seed=42)
# ds_train = ds_split["train"]
# ds_test = ds_split["test"]

# print(f"Dataset split: {len(ds_train)} train, {len(ds_test)} test samples.")

learning_rate = 1e-5

from unsloth import UnslothTrainer
from trl import SFTConfig

args = SFTConfig(
    dataset_text_field="text",
    output_dir=f"outputs/{product}",
    report_to="wandb",
    # packing=True,
    # max_steps=100,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    optim="paged_ademamix_8bit",
    # optim="adamw_8bit",
    learning_rate=learning_rate,
    lr_scheduler_type="polynomial",
    lr_scheduler_kwargs={"lr_end": 2e-6, "power": 1.0},
    # max_grad_norm=2,
    warmup_steps=100,
    # warmup_ratio=0.10,
    weight_decay=0.01,
    logging_steps=5,
    # eval_strategy="steps",
    # do_eval=True,
    # eval_steps=10,
    # save_strategy="steps",
    # save_steps = 100,
    # save_total_limit=3,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_split,
    # eval_dataset=ds_test,
    args=args,
)

# print the token-length of the first 10 samples in the dataset. those are already tokenized
for i in range(1000):
    length = len(trainer.train_dataset[i]["input_ids"])
    if length > 1024 * 5:
        print(f"Sample {i} token length: {length}")

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
# TODO only train on the last response in the conversation

import gc
import wandb

gc.collect()
torch.cuda.empty_cache()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project=product)
trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Write a story.",
            }
        ],
    }
]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
outputs = model.generate(
    **tokenizer([text], return_tensors="pt").to("cuda"),
    max_new_tokens=1024 * 4,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    frequency_penalty=1.0,
)
print(tokenizer.batch_decode(outputs))

# Save the model and tokenizer locally
model.save_pretrained(f"outputs/{product}")
tokenizer.save_pretrained(f"outputs/{product}")
# Online saving
model.push_to_hub(f"Nelathan/{product}")
tokenizer.push_to_hub(f"Nelathan/{product}")

print("Saved model")
