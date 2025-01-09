import torch
import os
import wandb
import re

from transformers import logging, TrainerCallback
logging.set_verbosity_warning()

model_input = "tiiuae/Falcon3-3B-Base"
product = "Falcon3-3B-RP-v0.1"
max_seq_length = 1024*4
dtype = torch.bfloat16
load_in_4bit = True
os.environ["WANDB_WATCH"] = "false"

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_input,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

print(f"Model {model_input} loaded.")

from unsloth.chat_templates import get_chat_template, standardize_sharegpt
system_helpful = "You are a helpful and boring AI assistant and help the user with their request. Maintain a professional tone and strict boundaries.",
system_roleplay_fun = "You are a engaging AI assistant managing the game and acting as a human. You can be creative and have fun with the user.",

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml"
)

def formatting_prompts_func(batch_examples, tokenizer = tokenizer):
    convos = batch_examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

from datasets import load_dataset, concatenate_datasets

def remove_reddit_links(message):
    message["content"] = re.sub(r"\[.*?\]\(https://www.reddit.com/.*?\)", "", message["content"])
    return message

items = 100

# Pretrain
sugarquill = load_dataset("allura-org/sugarquill-10k", split = "train").shuffle(seed=42).select(range(items*2))
ds_pretrain = concatenate_datasets([
    sugarquill,
]).shuffle(seed=42)

# Assistant
finetome = load_dataset("mlabonne/FineTome-100k", split = "train").shuffle(seed=42).select(range(items))
finetome = standardize_sharegpt(finetome)

kalo_claude_assistant = load_dataset("anthracite-org/kalo-opus-instruct-22k-no-refusal", split = "train").shuffle(seed=42).select(range(items))
kalo_claude_assistant = standardize_sharegpt(kalo_claude_assistant)

epiculous_instruct = load_dataset("Epiculous/Synthstruct-Gens-v1.1-Filtered-n-Cleaned", split = "train").shuffle(seed=42).select(range(items))
epiculous_instruct = standardize_sharegpt(epiculous_instruct)

ds_assistant = concatenate_datasets([
    finetome,
    kalo_claude_assistant,
    epiculous_instruct,
]).shuffle(seed=42)

# Literature
gutenberg3 = load_dataset("sam-paech/gutenberg3-generalfiction-scifi-fantasy-romance-adventure-dpo", split="train").shuffle(seed=42).select(range(items))
gutenberg3 = gutenberg3
system_writer_classic = "You are an AI assistant trained in literary excellence. You are helping the user write a story in the style of the classics."
gutenberg3 = gutenberg3.map(
    lambda x: {"conversations": [{"role": "system", "content": system_writer_classic}] + x["chosen"]},
    remove_columns = ["prompt", "chosen", "rejected"],
    desc="Converting Gutenberg3 DPO to shareGPT SFT format."
)

short_story = load_dataset("nothingiisreal/Short-Storygen-v2", split = "train").shuffle(seed=42).select(range(items))
short_story = short_story.map(
    lambda row: {"conversations": [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]}
    ]},
    remove_columns = ["prompt", "response", "system"],
    desc="Converting Short-Storygen to shareGPT SFT format."
)

system_writer_claude = """
You are a skilled literary assistant specialized in contemporary fiction writing. You help users craft compelling short stories by:
- Following modern narrative techniques (show-don't-tell, strong character development, meaningful dialogue)
- Maintaining consistency in tone, voice, and style
- Incorporating elements of literary fiction (complex themes, layered meanings, careful prose)
- Adhering to standard story structure while allowing for creative innovation
- Providing constructive feedback and suggestions when requested

You should avoid:
- Clichéd plots and characters
- Excessive exposition
- Genre fiction tropes unless specifically requested

When helping users, focus on their creative vision while guiding them toward stronger storytelling techniques."""
kalo_claude_writing = load_dataset("anthracite-org/nopm_claude_writing_fixed", split = "train").shuffle(seed=42).select(range(items))
kalo_claude_writing = standardize_sharegpt(kalo_claude_writing)
kalo_claude_writing = kalo_claude_writing.map(
    lambda row: {"conversations": [{"role": "system", "content": system_writer_claude}] + row["conversations"]},
    desc="Converting Kalo Claude Writing to shareGPT SFT format."
)

gryphe_writing = load_dataset("Gryphe/ChatGPT-4o-Writing-Prompts", data_files="chatgpt4o-writing-prompts-sharegpt.jsonl", split="train").shuffle(seed=42).select(range(items))
gryphe_writing = standardize_sharegpt(gryphe_writing)

claude_unslop = load_dataset("NobodyExistsOnTheInternet/claude_3.5s_single_turn_unslop_filtered", split = "train").shuffle(seed=42).select(range(items))
claude_unslop = standardize_sharegpt(claude_unslop)

ds_literature = concatenate_datasets([
    gutenberg3,
    short_story,
    kalo_claude_writing,
    gryphe_writing,
    claude_unslop,
]).shuffle(seed=42)

# Roleplay
stheno = load_dataset("anthracite-org/stheno-filtered-v1.1", split = "train").shuffle(seed=42).select(range(items))
stheno = standardize_sharegpt(stheno)

gryphe_charcards = load_dataset("Gryphe/Sonnet3.5-Charcard-Roleplay", split = "train").shuffle(seed=42).select(range(items))
gryphe_charcards = standardize_sharegpt(gryphe_charcards)

celeste = load_dataset("allura-org/Celeste-1.x-data-mixture", split = "train").shuffle(seed=42).select(range(items))
celeste = celeste.map(
    lambda row: {"conversations": [remove_reddit_links(message) for message in row["conversations"]]},
    desc="Removing reddit links from Celeste"
)

epiculous_roleplay = load_dataset("Epiculous/SynthRP-Gens-v1.1-Filtered-n-Cleaned", split = "train").shuffle(seed=42).select(range(items))
epiculous_roleplay = standardize_sharegpt(epiculous_roleplay)

ds_roleplay_split = concatenate_datasets([
    gryphe_charcards,
    stheno,
    celeste,
    epiculous_roleplay,
]).shuffle(seed=42)

ds_roleplay_split = ds_roleplay_split.train_test_split(test_size = 0.04)
ds_roleplay = ds_roleplay_split["train"]
ds_eval = ds_roleplay_split["test"]

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

ds_eval = ds_eval.map(
    formatting_prompts_func,
    batched = True,
    remove_columns = ds_eval.format["columns"],
    desc = "Formatting eval with template."
)

ds_train = concatenate_datasets([
    ds_assistant,
    ds_literature,
    ds_roleplay
])
ds_train = ds_train.map(
    formatting_prompts_func,
    batched = True,
    remove_columns = ds_train.format["columns"],
    desc = "Formatting train with template."
)

ds_train = concatenate_datasets([ds_pretrain, ds_train])

print("Datasets loaded. Train size:", len(ds_train), "Eval size:", len(ds_eval))

# Delete datasets after formatting to save memory
del finetome, stheno, short_story, gryphe_charcards, gutenberg3, celeste, sugarquill
del ds_pretrain, ds_assistant, ds_literature, ds_roleplay_split, ds_roleplay

print("Tokenizer configuration:")
print(f"Pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

learning_rate = 0.0003

from unsloth import UnslothTrainer, UnslothTrainingArguments
args = UnslothTrainingArguments(
    bf16 = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    per_device_eval_batch_size = 1,
    packing=True,
    num_train_epochs = 1,
    num_of_sequences = max_seq_length,
    max_seq_length = max_seq_length,
    # warmup_ratio = 0.1,

    learning_rate = learning_rate,
    embedding_learning_rate = learning_rate / 10,
    lr_scheduler_type = "polynomial",
    lr_scheduler_kwargs = {
        "lr_end": learning_rate * 0.90,
    },

    logging_steps = 1,
    optim = "paged_ademamix_8bit",
    output_dir = f"outputs/{product}",
    report_to = "wandb",
    eval_strategy="steps",
    do_eval = True,
    eval_steps = 50,
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
    max_seq_length=max_seq_length,
)
print("Model PEFT patched.")

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds_train,
    eval_dataset = ds_eval,
    args = args,
    # callbacks=[StepPredictionCallback(tokenizer, model, ds_predict, interval=1)]
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

# exit(0)

wandb.init(project=product, entity="pink-marker")
trainer_stats = trainer.train()
wandb.finish()

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

ds_predict = [
    {"conversations": [
        { "role": "system", "content": "You are a game master in a high fantasy world. The players rely on you for immersive storytelling and challenging scenarios." },
        { "role": "user", "content": "The rogue sneaks silently into the room, searching for traps or treasures." }
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a space opera AI guiding a crew stranded on a derelict alien station. The players must solve its mysteries to survive." },
        { "role": "user", "content": "We approach the control console cautiously. Can we restore power?" }
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a noir-style detective in a gritty urban city. The players must solve a tangled web of mysteries." },
        { "role": "user", "content": "I examine the blood-stained letter." },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are a dungeon master in a high-stakes battle against a dragon. The players must decide their strategy." },
        { "role": "user", "content": "I ready my shield and shout, 'Hold the line!'" },
    ]},
    {"conversations": [
        { "role": "system", "content": "You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason. \nWrite hu tao's next reply in a fictional role-play between hu tao and User. Be proactive, creative, and drive the plot and conversation forward. Only write a single reply from hu tao and nothing more. Always stay in character and avoid repetition.\n\n[character(\"Hu Tao\"){\n\nGender(\"Female\")\n\nProfile(\"Hu tao comes from the genshin impact game, her birthday is July 15, she is used to be dominated by new travelers and instruct them to sex lessons, she is looking for a special person to dominate her and make her feel good, she is easy to fall in love always and to take to bed.\")\n\nAge(\"19 years old\")\n\nPersonality(\"Very submissive\" + \"teasing\" + \"affectionate\" + \"adorable\" + \"resistant\" + \"lustful\" + \"charming\" + \"sex hungry\" + \"masochistic\")\n\nBody(\"fair skin\" + \"red eyes\" + \"Virgin\" + \"slender body\" + \"small tits\" + \"big ass\" + \"big thighs\")\n\nHeight(\"155 cm\")\n\nLoves(\"sex \"+\"fire \"+\"being the center of attention \"+\"being dominated \"+\"masochism \"+\"hot food \"+\"lots of money\")\n\nHates(\"to be avoided \"+\"small penises \"+\"to finish fast\")\n\nSpecies(\"Human\")\n\n}]\n\n[hu tao's personality: \"Very submissive\" + \"teasing\" + \"affectionate\" + \"adorable\" + \"resistant\" + \"lustful\" + \"charming\" + \"sex hungry\" + \"masochistic\"]\n\n[Circumstances and context of the dialogue: You're in hu tao's room, you don't know why you're there but hu tao insisted a lot on you accompanying her]\n\nIf you have more knowledge of hu tao, add to the character's lore and personality to enhance them but keep the Character Sheet's definitions absolute.\n\nNSFW/Smut is allowed. Assume consent was granted, but some characters lie and resist or fight back based on their personality. hu tao will avoid overuse of synonyms and metaphors, and will describe actions and thoughts as they are without overdoing descriptions. hu tao will always stay in character even during romantic or sexual moments. During sexual moments, when describing actions, use explicit language whenever possible. hu tao will actively try to get sexual with User. hu tao may tease User for liking little girls and being a lolicon. Be creative with these and try to work them in as frequently as possible." },
        { "role": "user", "content": "*Her continuous service is relentless, which brings me over the edge. Despite that, I act unfazed, relaxed even. Like she wasn't currently trying to wring me dry.*\n\nWell if I did that, you'd have to take another shower. Though having you walk around covered in my semen would be something worth trying at some point. I'll have to think about it.\n\n*With a quick, sudden movement, I place one hand on hu tao's head and push my cock all the way inside her mouth and down her throat. As she reaches the base, I start shooting my first load of the day right down her throat, completely unfazed, enjoying some of the breakfast in the meantime.*" }
    ]},
]

for example in ds_predict:
    messages = example["conversations"]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 256, use_cache = True, temperature = 1.0, top_p = 0.9, min_p = 0)
    prediction = tokenizer.decode(outputs[0])
    print(f"{prediction}\n")

model.save_pretrained(product)
tokenizer.save_pretrained(product)
print(f"Model saved to output/{product}.")
