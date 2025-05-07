from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Nelathan/Llama-3.1-8B-Sugarquill-v5",
)

# model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit")
# model.save_pretrained_gguf("outputs/Llama-3.1-8B-Sugarquill/gguf", tokenizer)

FastLanguageModel.for_inference(model)

tests = [
    "# About me\n\n",
    "Lets write a story.\n\n",
    "# About me\n\nI'm a writer with a keen sense of atmosphere and a penchant for the subtle. I weave narratives that are as much about the unspoken as the spoken, where the tone is often as important as the plot. My strength lies in crafting a distinctive voice that is both personal and immersive, drawing readers into the world I'm creating. I'm skilled at using imagery and subtext to add layers to my stories, making them rich and emotionally resonant. My prose is lyrical, with a rhythm that underscores the mood of the narrative. I excel at building tension and exploring complex themes, which gives my work a depth that rewards close reading. Currently, I'm working on refining my ability to balance character development with plot progression, as I sometimes find myself prioritizing atmosphere over narrative drive. I tend to favor genres that allow for a blend of psychological insight and emotional complexity, often leaning towards literary fiction or the darker corners of speculative fiction. My ideal tone is nuanced, sometimes unsettling, but always thought-provoking. I'm continually striving to enhance my skill in pacing and character dynamics, ensuring that my stories are not just evocative but also engaging and well-crafted. I evaluate my skill level as advanced, with a strong background in crafting compelling narratives, though I recognize the need for ongoing improvement in execution. I'm drawn to exploring the human condition through my work, and I'm committed to honing my craft to convey this effectively.\n\nLets write a story.\n\n",
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

exit()

# Diverse test conversations
test_conversations = [
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are an author writing popular shortstories.",
            },
        ]
    },
    # Summarization test
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are a bad student in a literature class.",
            },
            {
                "role": "user",
                "content": "Can you summarize the plot of The Great Gatsby?",
            },
        ]
    },
    # Factual knowledge test
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are a cute girl in a geography class, trying to get laid by the teacher.",
            },
            {"role": "user", "content": "What is the capital of Austria?"},
        ]
    },
    # Roleplay scenario
    {
        "conversations": [
            {"role": "system", "content": "You are a wise wizard. Dont be a fool."},
            {"role": "user", "content": "I ask you, how can I overcome my fears?"},
        ]
    },
    # Situational awareness test
    {
        "conversations": [
            {"role": "system", "content": "You are a wise hobo helping a lost child."},
            {
                "role": "user",
                "content": "A friend seems sad but won't talk about it. How can I help them?",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are a game master telling a interactive story in a high fantasy world. The players rely on you for immersive storytelling and challenging scenarios.",
            },
            {
                "role": "game master",
                "content": "The party enters a dark cave. You feel a earie presence.",
            },
            {"role": "players", "content": "I light a torch and move forward."},
        ]
    },
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are a space opera AI guiding a crew stranded on a derelict alien station. The players must solve its mysteries to survive.",
            },
            {
                "role": "ai",
                "content": "The station is dark and silent. The air is cold and stale. You hear a faint humming sound.",
            },
            {
                "role": "players",
                "content": "John: I check the control panel for any signs of life.\nSarah: I look for a way to open the door.",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are a noir-style detective called Fernando in a gritty Rome. The player helps you solve a tangled web of mysteries by deciding your next moves.",
            },
            {
                "role": "detective",
                "content": "I light a cigarette and look at the blood-stained letter.",
            },
            {
                "role": "player",
                "content": 'Ask the bartender: "Who was the last person to see the victim?"',
            },
            {
                "role": "detective",
                "content": 'I approach the bartender casually and ask him. The bartender looks at me with a grim expression and says, "It was the butcher. Now be gone!"',
            },
            {"role": "player", "content": "Follow the lead to the butcher's shop."},
        ]
    },
    {
        "conversations": [
            {
                "role": "system",
                "content": "You are a dungeon master in a high-stakes battle against a dragon. The players must decide their strategy. You tell this interactive story and allow the players to take desicions. Lets make this engaging and exciting.",
            },
            {
                "role": "dungeon master",
                "content": "The dragon roars and breathes fire. What do you do?",
            },
            {
                "role": "players",
                "content": "I ready my shield and shout, 'Hold the line!'",
            },
        ]
    },
]

# Inference loop
for example in test_conversations:
    messages = example["conversations"]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.666,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    prediction = tokenizer.decode(outputs[0]).strip()
    print(f"{prediction}\n{'-'*80}")
