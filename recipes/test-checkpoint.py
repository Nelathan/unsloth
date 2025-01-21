import torch

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/Llama-3.2-3B-Duck_20250116_221836/checkpoint-720",
    max_seq_length = 1024*4,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)
FastLanguageModel.for_inference(model)

model.save_pretrained_gguf("outputs/Llama-3.2-3B-Duck_20250116_221836/save", tokenizer, quantization_method = "q8_0")

# Diverse test conversations
test_conversations = [
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

    # Generate response with adjusted parameters
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        temperature=1,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # Decode and clean prediction
    prediction = tokenizer.decode(outputs[0]).strip()
    print(f"{prediction}\n{'-'*80}")
