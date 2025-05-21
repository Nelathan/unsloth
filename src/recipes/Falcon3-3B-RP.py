import torch
import os
import wandb
import re
from transformers import logging, PreTrainedTokenizerFast

logging.set_verbosity_warning()

model_input = "tiiuae/Falcon3-3B-Base"
product = "Falcon3-3B-Duck"
max_seq_length = 1024*4
dtype = torch.bfloat16
load_in_4bit = True
os.environ["WANDB_WATCH"] = "false"
packing = True

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_input,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

print(f"Model loaded.")

from unsloth.chat_templates import get_chat_template, standardize_sharegpt
system_helpful = "You are a helpful and boring AI assistant and help the user with their request. Maintain a professional tone and strict boundaries.",
system_roleplay_fun = "You are a engaging AI assistant managing the game and acting as a human. You can be creative and have fun with the user.",

freechatml_template = \
    "{% for message in messages %}"\
        "{{ '<|startoftext|>' + message['role'] + '\n' + message['content'] + '<|endoftext|>\n' }}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|startoftext|>' }}"\
    "{% endif %}"
pass

assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Invalid tokenizer type: {type(tokenizer)}"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = (freechatml_template, '<|endoftext|>')
)

tokenizer.add_special_tokens({
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
    "additional_special_tokens": [
        "<|startoftext|>",
    ]
}, True)

from datasets import load_dataset, concatenate_datasets

def remove_reddit_links(message):
    message["content"] = re.sub(r"\[.*?\]\(https://www.reddit.com/.*?\)", "", message["content"])
    return message

# 5k is 1h of training
minutes = 10
items = round(minutes / 60 * 5000 / 4)
# TODO merge literature and assistant

# Pretrain
sugarquill = load_dataset("allura-org/sugarquill-10k", split = "train")
system_sugarquill = "You are an author writing popular shortstories."
sugarquill = sugarquill.map(lambda row:
    { "conversations": [
        { "role": "system", "content": system_sugarquill },
        { "role": "assistant", "content": row["text"] }]
    }
)

# Assistant
airoboros = load_dataset("jondurbin/airoboros-3.2", split = "train")
airoboros = standardize_sharegpt(airoboros)
print(set([row["category"] for row in airoboros]))
airoboros = airoboros.filter(lambda row: row["category"] not in ["coding"])

# finetome = load_dataset("mlabonne/FineTome-100k", split = "train")
# finetome = standardize_sharegpt(finetome)

# kalo_claude_assistant = load_dataset("anthracite-org/kalo-opus-instruct-22k-no-refusal", split = "train")
# kalo_claude_assistant = standardize_sharegpt(kalo_claude_assistant)

# epiculous_instruct = load_dataset("Epiculous/Synthstruct-Gens-v1.1-Filtered-n-Cleaned", split = "train")
# epiculous_instruct = standardize_sharegpt(epiculous_instruct)

# Literature
system_writer_classic = "You are an AI assistant trained in literary excellence. You are helping the user write a story in the style of the classics."
system_writer_claude = "You are a skilled literary assistant specialized in contemporary fiction writing. You help users craft compelling short stories."

# gutenberg1 = load_dataset("jondurbin/gutenberg-dpo-v0.1", split="train")
# gutenberg1 = gutenberg1.map(
#     lambda row: {
#         "conversations": [
#             {"role": "system", "content": system_writer_classic},
#             {"role": "user", "content": row["prompt"]},
#             {"role": "assistant", "content": row["chosen"]}
#         ]
#     },
#     remove_columns = ["prompt", "chosen", "rejected", "rejected_model"],
#     desc="Converting Gutenberg1 DPO to shareGPT SFT format."
# )

# gutenberg2 = load_dataset("nbeerbower/gutenberg-moderne-dpo", split="train")
# gutenberg2 = gutenberg2.map(
#     lambda row: {
#         "conversations": [
#             {"role": "system", "content": row["prompt"]},
#             {"role": "user", "content": row["summary"]},
#             {"role": "assistant", "content": row["chosen"]}
#         ]
#     },
#     remove_columns = ["prompt", "chosen", "rejected", "summary"],
#     desc="Converting Gutenberg2 DPO to shareGPT SFT format."
# )

# gutenberg3 = load_dataset("sam-paech/gutenberg3-generalfiction-scifi-fantasy-romance-adventure-dpo", split="train")
# gutenberg3 = gutenberg3.map(
#     lambda x: {"conversations": [{"role": "system", "content": system_writer_classic}] + x["chosen"]},
#     remove_columns = ["prompt", "chosen", "rejected"],
#     desc="Converting Gutenberg3 DPO to shareGPT SFT format."
# )
# print("Gutenberg3 sample:", gutenberg3[0]["conversations"])

# short_story = load_dataset("nothingiisreal/Short-Storygen-v2", split = "train")
# short_story = short_story.map(
#     lambda row: {"conversations": [
#         {"role": "system", "content": row["system"]},
#         {"role": "user", "content": row["prompt"]},
#         {"role": "assistant", "content": row["response"]}
#     ]},
#     remove_columns = ["prompt", "response", "system"],
#     desc="Converting Short-Storygen to shareGPT SFT format."
# )

# kalo_claude_writing = load_dataset("anthracite-org/nopm_claude_writing_fixed", split = "train")
# kalo_claude_writing = standardize_sharegpt(kalo_claude_writing)
# kalo_claude_writing = kalo_claude_writing.map(
#     lambda row: {"conversations": [{"role": "system", "content": system_writer_claude}] + row["conversations"]},
#     desc="Converting Kalo Claude Writing to shareGPT SFT format."
# )

# gryphe_writing = load_dataset("Gryphe/ChatGPT-4o-Writing-Prompts", data_files="chatgpt4o-writing-prompts-sharegpt.jsonl", split="train")
# gryphe_writing = standardize_sharegpt(gryphe_writing)

# claude_unslop = load_dataset("NobodyExistsOnTheInternet/claude_3.5s_single_turn_unslop_filtered", split = "train")
# claude_unslop = standardize_sharegpt(claude_unslop)

# Roleplay
# stheno = load_dataset("anthracite-org/stheno-filtered-v1.1", split = "train")
# stheno = standardize_sharegpt(stheno)

# gryphe_charcards = load_dataset("Gryphe/Sonnet3.5-Charcard-Roleplay", split = "train")
# gryphe_charcards = standardize_sharegpt(gryphe_charcards)

# celeste = load_dataset("allura-org/Celeste-1.x-data-mixture", split = "train")
# celeste = celeste.map(
#     lambda row: {"conversations": [remove_reddit_links(message) for message in row["conversations"]]},
#     desc="Removing reddit links from Celeste"
# )

# epiculous_roleplay = load_dataset("Epiculous/SynthRP-Gens-v1.1-Filtered-n-Cleaned", split = "train")
# epiculous_roleplay = standardize_sharegpt(epiculous_roleplay)

# ds_roleplay_split = concatenate_datasets([
#     gryphe_charcards,
#     stheno,
#     celeste,
#     epiculous_roleplay,
# ]).shuffle(seed=42).select(range(items))

# ds_dict = {
#     "finetome": finetome,
#     "short_story": short_story,
#     "stheno": stheno,
#     "gryphe_charcards": gryphe_charcards,
#     "gutenberg3": gutenberg3,
#     "celeste": celeste,
#     "sugarquill": sugarquill,
# }

# for (name, dataset) in ds_dict.items():
#     print(f"Dataset: {name}")
#     print(f"Features: {dataset.features}")
#     print(f"Shape: {dataset.shape}")
#     print()

def formatting_prompts_func(batch_examples, tokenizer = tokenizer):
    convos = batch_examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

ds_split = concatenate_datasets([
    airoboros,
    sugarquill
])
ds_split = ds_split.map(
    formatting_prompts_func,
    remove_columns=ds_split.column_names,
    batched = True,
    desc="Formatting Training"
)

ds_split = ds_split.train_test_split(test_size=200, seed=42)
ds_train = ds_split["train"]
ds_eval = ds_split["test"]

print("Datasets loaded. Train size:", len(ds_train), "Eval size:", len(ds_eval), "\n")

# Delete datasets after formatting to save memory
del sugarquill, airoboros

learning_rate = 5e-5

from unsloth import UnslothTrainer, UnslothTrainingArguments
from datetime import datetime
timetamp = datetime.now().strftime('%Y%m%d_%H%M%S')

args = UnslothTrainingArguments(
    output_dir = "outputs/" + product + f"_{timetamp}",
    run_name = f"{timetamp}",
    report_to = "wandb",
    bf16 = True,

    packing=True,
    logging_steps = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    max_seq_length = max_seq_length,

    optim = "paged_ademamix_8bit",
    learning_rate = learning_rate,
    embedding_learning_rate = learning_rate * 0.15,
    lr_scheduler_type = "polynomial",
    lr_scheduler_kwargs = { "lr_end": learning_rate * 0.60, "power": 1.0 },
    warmup_ratio = 0.1,
    # warmup_steps = 10,

    eval_strategy="steps",
    do_eval = True,
    eval_steps = 40,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    lora_alpha = 128,
    target_modules = [
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
        "lm_head", "embed_tokens",
    ],
    use_gradient_checkpointing="unsloth",
    max_seq_length=max_seq_length,
)
print("Model PEFT patched.")

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds_train,
    eval_dataset = ds_eval,
    args = args,
)
print("Trainer created successfully.")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project=product, entity="pink-marker", save_code=True)
from unsloth import unsloth_train
# trainer_stats = trainer.train() << Buggy gradient accumulation
trainer_stats = unsloth_train(trainer)

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

FastLanguageModel.for_inference(model)

test_conversations = [
    {"conversations": [
        { "role": "system", "content": "You are a game master telling a interactive story in a high fantasy world. The players rely on you for immersive storytelling and challenging scenarios." },
        { "role": "game master", "content": "The party enters a dark cave. You feel a earie presence." },
        { "role": "players", "content": "I light a torch and move forward." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a space opera AI guiding a crew stranded on a derelict alien station. The players must solve its mysteries to survive." },
        { "role": "ai", "content": "The station is dark and silent. The air is cold and stale. You hear a faint humming sound." },
        { "role": "players", "content": "John: I check the control panel for any signs of life.\nSarah: I look for a way to open the door." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a noir-style detective called Fernando in a gritty Rome. The player helps you solve a tangled web of mysteries by deciding your next moves." },
        { "role": "detective", "content": "I light a cigarette and look at the blood-stained letter." },
        { "role": "player", "content": "Ask the bartender: \"Who was the last person to see the victim?\"" },
        { "role": "detective", "content": "I approach the bartender casually and ask him. The bartender looks at me with a grim expression and says, \"It was the butcher. Now be gone!\"" },
        { "role": "player", "content": "Follow the lead to the butcher's shop." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a dungeon master in a high-stakes battle against a dragon. The players must decide their strategy. You tell this interactive story and allow the players to take desicions. Lets make this engaging and exciting." },
        { "role": "dungeon master", "content": "The dragon roars and breathes fire. What do you do?" },
        { "role": "players", "content": "I ready my shield and shout, 'Hold the line!'" },
    ]},
]

test_conversations += ds_eval.shuffle(seed=42).select(range(5)).map(lambda row: {
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
        temperature = 1.0,
        top_p = 0.9,
        min_p = 0,
        repetition_penalty = 1.1,
    )
    prediction = tokenizer.decode(outputs[0])
    print(f"{prediction}\n")

model.save_pretrained(product)
tokenizer.save_pretrained(product)
print(f"Model saved to outputs/{product}_{timetamp}")
