import torch
import os
import wandb

from transformers import logging, PreTrainedTokenizerFast
logging.set_verbosity_warning()

model_input = "tiiuae/Falcon3-3B-Base"
product = "Falcon3-3B-Duck"
max_seq_length = 1024*8
dtype = torch.bfloat16
load_in_4bit = True

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_input,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("Model loaded successfully.")

assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Invalid tokenizer type: {type(tokenizer)}"

freechatml_template = \
    "{% for message in messages %}"\
        "{{ '<|startoftext|>' + message['role'] + '\n' + message['content'] + '<|endoftext|>\n' }}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|startoftext|>' }}"\
    "{% endif %}"
pass

from unsloth.chat_templates import get_chat_template, standardize_sharegpt

assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Invalid tokenizer type: {type(tokenizer)}"
tokenizer.padding_side = "left"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = (freechatml_template, '<|endoftext|>')
)

model.resize_token_embeddings(len(tokenizer))

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }
pass

from datasets import load_dataset, concatenate_datasets

# airoboros = load_dataset("jondurbin/airoboros-3.2", split = "train")
# airoboros = standardize_sharegpt(airoboros)

sugarquill = load_dataset("allura-org/sugarquill-10k", split = "train")
system_sugarquill = "You are an author writing popular shortstories."
sugarquill = sugarquill.map(lambda row:
    { "conversations": [
        { "role": "system", "content": system_sugarquill },
        { "role": "author", "content": row["text"] }]
    }
)

ds_split = concatenate_datasets([sugarquill])

ds_split = ds_split.map(
    formatting_prompts_func,
    remove_columns=ds_split.column_names,
    batched = True,
    desc="Formatting Training"
)

ds_split = ds_split.train_test_split(200, seed=42)
ds_train = ds_split["train"]
ds_test = ds_split["test"]

print(f"Dataset split: {len(ds_train)} train, {len(ds_test)} test samples.")

learning_rate = 100e-5

from unsloth import UnslothTrainer, UnslothTrainingArguments
from datetime import datetime
from trl import SFTConfig

timetamp = datetime.now().strftime('%Y%m%d_%H%M%S')
args = SFTConfig(
    output_dir = "outputs/" + product + f"/{timetamp}",
    run_name = f"{timetamp}",
    report_to = "wandb",
    bf16 = True,

    # packing=True,
    num_train_epochs = 1,
    max_seq_length = max_seq_length,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = 1,

    optim = "paged_ademamix_8bit",
    learning_rate = learning_rate,
    lr_scheduler_type = "polynomial",
    lr_scheduler_kwargs = { "lr_end": learning_rate * 0.70, "power": 1.0 },
    max_grad_norm = 2.0,
    warmup_steps = 100,
    # warmup_ratio = 0.05,
    # weight_decay = 0.05,

    logging_steps = 10,
    eval_strategy="steps",
    do_eval = True,
    eval_steps = 100,
    save_strategy = 'steps',
    # save_steps = 100,
    save_total_limit=3,
)

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
print(">>> LoRA weights added")

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds_train,
    eval_dataset = ds_test,
    args = UnslothTrainingArguments(
        embedding_learning_rate = learning_rate * 0.1,
        **args.to_dict()
    )
)
print(">>> Trainer created successfully.")

for i, batch in enumerate(trainer.get_train_dataloader()):
    if i > 0:
        break
    print(f"Batch {i+1} labels:")
    for row in batch["labels"]:
        labels = row.clone()
        labels[labels == -100] = tokenizer.pad_token_id
        print(tokenizer.decode(labels), "\n")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

import gc
gc.collect()
torch.cuda.empty_cache()

wandb.init(project=product)
# from unsloth import unsloth_train
# trainer_stats = unsloth_train(trainer)
print(">>> START")

trainer_stats = trainer.train()


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

model.save_pretrained(product)
tokenizer.save_pretrained(product)
print(f"Model saved to output/{product}.")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

test_conversations = [
    {"conversations": [
        { "role": "system", "content": "You are an author writing popular shortstories." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a game master telling a interactive story in a high fantasy world. The users rely on you for immersive storytelling and challenging scenarios." },
        { "role": "assistant", "content": "The party enters a dark cave. You feel a earie presence." },
        { "role": "user", "content": "I light a torch and move forward." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a space opera AI guiding a crew stranded on a derelict alien station. The user must solve its mysteries to survive." },
        { "role": "assistant", "content": "The station is dark and silent. The air is cold and stale. You hear a faint humming sound." },
        { "role": "user", "content": "John: I check the control panel for any signs of life.\nSarah: I look for a way to open the door." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a noir-style detective called Fernando in a gritty Rome. The user helps you solve a tangled web of mysteries by deciding your next moves." },
        { "role": "assistant", "content": "I light a cigarette and look at the blood-stained letter." },
        { "role": "user", "content": "Ask the bartender: \"Who was the last person to see the victim?\"" },
        { "role": "assistant", "content": "I approach the bartender casually and ask him. The bartender looks at me with a grim expression and says, \"It was the butcher. Now be gone!\"" },
        { "role": "user", "content": "Follow the lead to the butcher's shop." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a dungeon master in a high-stakes battle against a dragon. The user must decide their strategy. You tell this interactive story and allow the user to take desicions. Lets make this engaging and exciting." },
        { "role": "assistant", "content": "The dragon roars and breathes fire. What do you do?" },
        { "role": "user", "content": "I ready my shield and shout, 'Hold the line!'" },
    ]},
]

test_conversations += ds_test.shuffle(seed=42).select(range(5)).map(lambda row: {
    # drop the last message, as it is the prompt
    "conversations": row["conversations"][:-1]
})

for example in test_conversations:
    messages = example["conversations"]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_new_tokens = 1024,
        temperature = 0.666,
        top_p = 0.9,
        min_p = 0,
        repetition_penalty = 1.1,
        # no_repeat_ngram_size = 4,
    )
    prediction = tokenizer.decode(outputs[0])
    # this is the whole conversation, including the prompt
    print(f"{prediction}\n")
