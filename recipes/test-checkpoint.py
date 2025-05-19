from unsloth import FastLanguageModel
import codecs
import random

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Nelathan/Llama-3.1-8B-Sugarquill-v13",
)

# model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit")
# model.save_pretrained_gguf("outputs/Llama-3.1-8B-Sugarquill/gguf", tokenizer)

FastLanguageModel.for_inference(model)

test_story_profile = """# About me

I craft narratives designed to resonate deeply, immersing readers in worlds where each element feels both intentional and effortlessly woven. My prose is honed for clarity, impact, and a natural rhythm, evoking vivid imagery and profound emotional depths, all carried by a distinct voice rich with subtext. I thrive on building palpable tension and exploring intricate character dynamics, guiding individuals through meaningful arcs that illuminate complex, universal themes. My aim is always to offer fresh perspectives within a coherent structure, ensuring transformations are earned and the experience lingers, sparking thought long after the story concludes. I am drawn to the nuanced exploration of the human condition, often through a literary lens, seeking to create connections that feel both authentic and unforgettable."""

test_story_notes = """
Okay, my private ledger for this new piece...

**Working Title:** The Alabaster Echo

**Logline:** A horologist, obsessed with a single, unrecoverable moment from his past, discovers a peculiar resonance in antique clockworks that seems to offer a fleeting, distorted return, forcing him to confront whether memory preserved is life lived, or life paused.

**Core Question:** At what point does remembrance become a gilded cage, and clarity a cruel mirage?

**Protagonist:** Elias Thorne, mid-40s. A craftsman of meticulous habit, his grief for his late wife, Clara, has calcified into an almost monastic devotion to his workshop and the precise mechanics of time. His voice is quiet, his observations keen, his sorrow a polished stone he turns over and over.

**Inciting Incident (The Seed):** Elias acquires a rare, non-functional 18th-century automaton clock. While attempting its restoration, a specific combination of chimes and mechanical whirrings triggers an intensely vivid, almost tactile memory-fragment of Clara – clearer, more *present* than anything he's experienced in years. It's disorienting, addictive.

**Thematic Resonance (The Undertow):**
*   The nature of memory: its malleability, its comfort, its tyranny.
*   Grief as a temporal distortion.
*   The beauty and danger of obsession.
*   The illusion of control vs. the acceptance of loss.

**Key Sensory Focus:**
*   **Sound:** The ticking, whirring, chiming of clocks. The subtle sigh of old wood. The remembered cadence of Clara's voice. Silence.
*   **Touch:** The coldness of metal, the smooth wear of old tools, the imagined warmth of a hand.
*   **Sight:** Dust motes in workshop light, the intricate dance of gears, the faded photograph of Clara that is his current, insufficient touchstone.

**Plot Beats & Tension Arc (The Escapement):**
1.  **Initial Resonance:** The first accidental trigger. Joy, confusion, a desperate need to replicate it.
2.  **The Pursuit:** Elias meticulously experiments with the automaton and other old clocks, seeking the "frequency" of that memory. Days blur. His workshop becomes a sanctuary and a laboratory.
3.  **Fleeting Successes, Deepening Immersion:** He manages to evoke further fragments. Each is a potent hit, but they are slightly *off*. A different scent, a misremembered word. The edges blur between the echo and his own desperate desire. Subtly, his present-day interactions (the few he has – a concerned apprentice, perhaps, or a nosy neighbour) become tinged with distraction, impatience.
4.  **The Price of Clarity (The Subtextual Bleed):** The "echoes" begin to subtly alter his *present* perception of Clara. The cherished, static photograph on his workbench starts to feel… less accurate. The more he "visits," the more the authentic memory seems to erode, replaced by these compelling, yet slightly fabricated, visitations. He might start misremembering details he was once certain of outside the echo-state.
5.  **The Apprentice's Concern (Character Dynamic):** Young Thomas, his apprentice, notices Elias's decline – the unkempt workshop, the missed appointments, the vacant stares. Thomas represents the pull of the present, of life continuing. His attempts to reach Elias create gentle friction.
6.  **Climax (The Unwinding Spring):** Elias, in a desperate attempt to fully reconstruct *the* moment (perhaps their last good day, or a moment of profound connection), pushes the automaton (and himself) too far. The resulting echo is vivid, prolonged, but horribly distorted. Clara is there, but she says something cruel, or her face is a stranger's, or the setting is nightmarish. The illusion shatters, revealing the hollow core of his pursuit. He realizes he's been polishing a phantom, sacrificing the true, albeit imperfect, memory.
7.  **Resolution (The Pendulum Settles):** Elias, shaken, steps away from the automaton. He might not destroy it, but he covers it. He looks at Clara's photograph, truly *sees* its imperfections, and finds a quiet, melancholic acceptance. The grief is still there, but the frantic need to *relive* has been replaced by a more grounded, tender remembrance. Perhaps he engages with Thomas properly for the first time in weeks, a small gesture towards rejoining the flow of time. The final image could be him opening the workshop window, letting in fresh air and the sounds of the outside world. No grand epiphanies, but a subtle, earned shift.

**Voice & Prose Notes (The Caliber):**
*   Internal, reflective, elegiac.
*   Precision in describing mechanical processes, contrasted with the elusive nature of memory.
*   Imagery should be sharp but imbued with emotional weight.
*   Rhythm: measured, allowing moments of quiet intensity to breathe.

**Avoidances (Checks & Balances):**
*   No sudden, unearned transformations. Elias's shift must be gradual, painful.
*   No melodrama. The emotion is deep, but expressed with restraint.
*   The "mechanism" of the memory echo remains ambiguous, almost mystical – no need for pseudo-scientific exposition. It's a catalyst, not the focus.
*   Keep supporting characters functional but not caricatures. Thomas is concern, not a plot device.

This framework feels right. It offers scope for the finesse I value, while ensuring a solid core and meaningful execution. The inherent mystery isn't "whodunnit" but "what is happening to memory, to Elias?" That's the vein I want to tap.
"""

user_names = [
    "Alex",
    "Jordan",
    "Taylor",
    "Morgan",
    "Casey",
    "Riley",
    "Sam",
    "Jamie",
    "Avery",
    "Quinn",
    "Drew",
    "Skyler",
]

user_prompts = [
    "I'd really enjoy reading a story from you.",
    "Stories from you always brighten my day.",
    "If you have a story in mind, I'd love to hear it.",
    "Whenever you feel inspired, a story would be wonderful.",
    "Your storytelling always inspires me.",
    "I appreciate your creativity—share a story when you can.",
    "A story from you would be a treat.",
    "I'm in the mood for something imaginative, if you're up for it.",
    "Your stories always make me think.",
    "If you're feeling creative, I'd enjoy a story.",
    "I always look forward to your stories.",
    "Whenever you're ready, I'd love to read something new.",
]

# format as a chat
tests = [
    [
        {
            "role": random.choice(user_names),
            "content": random.choice(user_prompts),
        },
    ],
    [
        {"role": "author", "content": test_story_profile},
        {
            "role": random.choice(user_names),
            "content": random.choice(user_prompts),
        },
    ],
    [
        {"role": "author", "content": test_story_profile},
        {
            "role": random.choice(user_names),
            "content": "Plan a story, then write it.",
        },
    ],
]

# transform the test cases to the chat template
for i in range(len(tests)):
    tests[i] = tokenizer.apply_chat_template(
        tests[i], tokenize=False, add_generation_prompt=True
    )

# for the last test case, add the notes
tests[-1] = (
    tests[-1]
    + "author<|end_header_id|>\n\n<THINKING>\n"
    + test_story_notes
    + "\n</THINKING>\n\n# The Alabaster Echo\n\n"
)

for text in tests:
    # tokenize
    rec = tokenizer(text, return_tensors="pt").to("cuda")
    # generate
    outputs = model.generate(
        **rec,
        max_new_tokens=1024 * 8,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=[
            tokenizer.eos_token_id,
            128009,  # is the end of chat message token: <|eot_id|>
        ],
    )
    outputs = tokenizer.batch_decode(outputs)
    for output in outputs:
        clean = output.replace("\\n", "\n").replace("\\t", "\t")
        print(clean)
    print("===" * 16)
