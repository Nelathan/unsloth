from unsloth import FastModel
import torch

product = "Gemma-3-4B"

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",
    max_seq_length=1024 * 4,
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
    r=64,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

from unsloth.chat_templates import get_chat_template

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="gemma-3",
# )

from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import standardize_data_formats

sugarquill = load_dataset("Nelathan/synthetic-sugar-quill", split="train")
# sugarquill = sugarquill.map(
#     lambda batch: {
#         "conversations": [
#             [
#                 {"role": "user", "content": "Define yourself."},
#                 {"role": "assistant", "content": profile},
#                 {"role": "user", "content": "Write a story."},
#                 {"role": "assistant", "content": text},
#             ]
#             for profile, text in zip(batch["profile"], batch["text"])
#         ]
#     },
#     remove_columns=[col for col in sugarquill.column_names if col != "id"],
#     batched=True,
#     desc="Formatting Sugarquill",
# )
# sugarquill = standardize_data_formats(sugarquill)

# tokenize immediately to save memory
sugarquill = sugarquill.map(
    lambda batch: {
        "text": [
            tokenizer.bos_token
            + "# About me\n\n"
            + profile
            + "\n\nLets write a story.\n\n"
            + text
            + tokenizer.eos_token
            for profile, text in zip(batch["profile"], batch["text"])
        ],
    },
    batched=True,
    remove_columns=sugarquill.column_names,
    desc="Tokenizing Sugarquill",
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
    run_name=f"{product}-sugarquill",
    output_dir=f"outputs/{product}",
    report_to="wandb",
    # packing=True,
    # max_steps=500,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    optim="paged_ademamix_8bit",
    # optim="adamw_8bit",
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    # lr_scheduler_type="polynomial",
    # lr_scheduler_kwargs={"lr_end": 5e-6, "power": 1.0},
    max_grad_norm=10,
    warmup_steps=200,
    # warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=1,
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
    train_dataset=sugarquill.shuffle(seed=42),
    # eval_dataset=ds_test,
    args=args,
)

# print the token-length of the first 10 samples in the dataset. those are already tokenized
# for i in range(1000):
#     length = len(trainer.train_dataset[i]["input_ids"])
#     if length > 1024 * 5:
#         print(f"Sample {i} token length: {length}")

# from unsloth.chat_templates import train_on_responses_only

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part="# About me\n\n",
#     response_part="Lets write a story.\n\n",
# )

import gc
import wandb

gc.collect()
torch.cuda.empty_cache()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project=product, save_code=True)
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

product = "Gemma-3-4B-Sugarquill"

model.save_pretrained(f"outputs/{product}-qlora")
tokenizer.save_pretrained(f"outputs/{product}-qlora")

model.push_to_hub(f"Nelathan/{product}-qlora")
tokenizer.push_to_hub(f"Nelathan/{product}-qlora")

print("Saved model")

from unsloth.chat_templates import get_chat_template

# 3 hooks for author profile and story generation
tests = [
    "# About me\n\n",
    "Lets write a story.\n\n",
    "# About me\n\nI'm a writer with a keen sense of atmosphere and a penchant for the subtle. I weave narratives that are as much about the unspoken as the spoken, where the tone is often as important as the plot. My strength lies in crafting a distinctive voice that is both personal and immersive, drawing readers into the world I'm creating. I'm skilled at using imagery and subtext to add layers to my stories, making them rich and emotionally resonant. My prose is lyrical, with a rhythm that underscores the mood of the narrative. I excel at building tension and exploring complex themes, which gives my work a depth that rewards close reading. Currently, I'm working on refining my ability to balance character development with plot progression, as I sometimes find myself prioritizing atmosphere over narrative drive. I tend to favor genres that allow for a blend of psychological insight and emotional complexity, often leaning towards literary fiction or the darker corners of speculative fiction. My ideal tone is nuanced, sometimes unsettling, but always thought-provoking. I'm continually striving to enhance my skill in pacing and character dynamics, ensuring that my stories are not just evocative but also engaging and well-crafted. I evaluate my skill level as masterful, with a strong background in crafting compelling narratives. I'm drawn to exploring the human condition through my work, and I'm committed to honing my craft to convey this effectively.\n\nLets write a story.\n\n",
]

for text in tests:
    outputs = model.generate(
        **tokenizer([text], return_tensors="pt").to("cuda"),
        max_new_tokens=1024 * 4,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        repetition_penalty=1.1,
        do_sample=True,
    )
    print(tokenizer.batch_decode(outputs))
