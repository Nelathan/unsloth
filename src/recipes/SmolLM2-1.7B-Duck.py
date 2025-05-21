import torch
import os
import wandb

from transformers import logging, PreTrainedTokenizerFast
logging.set_verbosity_warning()

model_input = "unsloth/SmolLM2-1.7B"
product = "SmolLM2-1.7B-Duck"
max_seq_length = 1024*4
dtype = torch.bfloat16
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
os.environ["WANDB_WATCH"] = "false"  # Disable gradient logging

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_input,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)
print("Model loaded successfully.")

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
)
print("Model patched successfully.")

print("tokenizer", tokenizer)

freechatml_template = \
    "{% for message in messages %}"\
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|endoftext|>\n' }}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|im_start|>' }}"\
    "{% endif %}"
pass

from unsloth.chat_templates import get_chat_template, standardize_sharegpt
tokenizer = get_chat_template(
    tokenizer,
    chat_template = (freechatml_template, '<|endoftext|>'),
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }
pass

from datasets import load_dataset, concatenate_datasets

sugarquill = load_dataset("allura-org/sugarquill-10k", split = "train")
system_sugarquill = "You are an author writing popular shortstories."
sugarquill = sugarquill.map(lambda row:
    { "conversations": [
        { "role": "system", "content": system_sugarquill },
        { "role": "author", "content": row["text"] }]
    }
)

# finetome = load_dataset("mlabonne/FineTome-100k", split = "train")
# finetome = standardize_sharegpt(finetome)

# finetome = finetome.map(lambda row:
#     { "length": sum([len(message["content"]) for message in row["conversations"]]) }
# )

airoboros = load_dataset("jondurbin/airoboros-3.2", split = "train")
airoboros = standardize_sharegpt(airoboros)
airoboros = airoboros.filter(lambda row: row["category"] not in ["coding", "math"])

# airoboros = airoboros.map(lambda row:
#     { "length": sum([len(message["content"]) for message in row["conversations"]]) }
# )

# import numpy as np
# lengths = np.array([row["length"] for row in finetome])
# bins = np.linspace(0, 8000, 40)
# histogram, bin_edges = np.histogram(lengths, bins=bins)
# print("Histogram of conversation lengths:")
# for i in range(len(histogram)):
#     # respecting the total size, scaling the histogram to 100 characters
#     print(f"{int(bin_edges[i]):4d}-{int(bin_edges[i+1]):4d}: {'#' * int(histogram[i] / max(histogram) * 100)}")
# pass

# thus dropping all long conversations
# finetome = finetome.sort('length').select(range(20000))
# airoboros = airoboros.sort('length')\
    # .select(range(50000))

# celeste = load_dataset("allura-org/Celeste-1.x-data-mixture", split = "train[:400]")

ds_pre = sugarquill.map(
    formatting_prompts_func,
    remove_columns=sugarquill.column_names,
    batched = True,
    desc="Formatting Pre"
)
ds_sft = airoboros.map(
    formatting_prompts_func,
    remove_columns=airoboros.column_names,
    batched = True,
    desc="Formatting SFT"
)

ds_train = concatenate_datasets([
    ds_pre.shuffle(seed=42),
    ds_sft.shuffle(seed=42)
])

ds_split = ds_train.train_test_split(test_size=200, seed=42)
ds_train = ds_split["train"]
ds_test = ds_split["test"]

print(f"Dataset split: {len(ds_train)} train, {len(ds_test)} test samples.")

# test if any property "text" is None
for i, row in enumerate(ds_train):
    if row["text"] is None:
        print("error in row:", i)
        break
pass

learning_rate = 5e-5
from unsloth import UnslothTrainer, UnslothTrainingArguments
args = UnslothTrainingArguments(
    output_dir = "outputs",
    report_to = "wandb",
    bf16 = True,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = 2,
    logging_steps = 1,

    packing=True,
    num_train_epochs = 1,
    max_seq_length = max_seq_length,

    optim = "paged_ademamix_8bit",
    learning_rate = learning_rate,
    embedding_learning_rate = learning_rate * 0.15,
    lr_scheduler_type = "polynomial",
    lr_scheduler_kwargs = { "lr_end": learning_rate * 0.60, "power": 1.0 },
    # warmup_steps = 50,
    # max_grad_norm = 0.5,
    warmup_ratio = 0.1,
    # max_steps = 100,

    eval_strategy="steps",
    do_eval = True,
    eval_steps = 25,
)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds_train.shuffle(seed=42),
    eval_dataset = ds_test,
    args = args
)
print("Trainer created")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project=product, entity="pink-marker", save_code=True)
from unsloth import unsloth_train
# trainer_stats = trainer.train() << Buggy gradient accumulation
trainer_stats = unsloth_train(trainer)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

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

# actually use out of distribution data
test_conversations += airoboros.shuffle(seed=42).select(range(5)).map(lambda row: {
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
        # no_repeat_ngram_size = 4,
    )
    prediction = tokenizer.decode(outputs[0])
    print(f"{prediction}\n")

model.save_pretrained(product)
tokenizer.save_pretrained(product)
print(f"Model saved to output/{product}.")
