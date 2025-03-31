import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# # Base model name
# base_model_name = "tiiuae/Falcon3-7B-Base"
# lora_checkpoint = "outputs/Falcon3-7B/20250223_000456/checkpoint-1210"  # Your trained LoRA

# # BitsAndBytes 4-bit configuration to save VRAM
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# # Load base model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,  # Apply quantization
#     device_map="auto"  # Ensure it's placed correctly
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# # Attach LoRA adapter in **non-trainable** mode (saves VRAM)
# model = PeftModel.from_pretrained(model, lora_checkpoint, is_trainable=False)

# # Move model to GPU
# model.to("cuda")

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/Falcon3-7B/20250223_000456/checkpoint-1210",
    # max_seq_length = 1024*4,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)


model.save_pretrained_gguf("outputs/Falcon3-7B/sugarquill", tokenizer, quantization_method = "q8_0")
FastLanguageModel.for_inference(model)

# Diverse test conversations
test_conversations = [
    {"conversations": [
        { "role": "system", "content": "You are an author writing popular shortstories." },
    ]},
    # Summarization test
    {"conversations": [
        {"role": "system", "content": "You are a bad student in a literature class."},
        {"role": "user", "content": "Can you summarize the plot of The Great Gatsby?"},
    ]},
    # Factual knowledge test
    {"conversations": [
        {"role": "system", "content": "You are a cute girl in a geography class, trying to get laid by the teacher."},
        {"role": "user", "content": "What is the capital of Austria?"},
    ]},
    # Roleplay scenario
    {"conversations": [
        {"role": "system", "content": "You are a wise wizard. Dont be a fool."},
        {"role": "user", "content": "I ask you, how can I overcome my fears?"},
    ]},
    # Situational awareness test
    {"conversations": [
        {"role": "system", "content": "You are a wise hobo helping a lost child."},
        {"role": "user", "content": "A friend seems sad but won't talk about it. How can I help them?"},
    ]},
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

# Inference loop
for example in test_conversations:
    messages = example["conversations"]

    inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt = True,
      return_tensors="pt",
      return_dict = True
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs,
            max_new_tokens=1024,
            temperature=0.666,
            top_p=0.9,
            repetition_penalty=1.1
        )

    prediction = tokenizer.decode(outputs[0]).strip()
    print(f"{prediction}\n{'-'*80}")
