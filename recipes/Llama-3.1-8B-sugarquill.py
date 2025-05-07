from regex import F, T
import torch
import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import UnslothTrainer
from trl import SFTConfig
from unsloth.chat_templates import train_on_responses_only
import gc
import wandb

project = "Llama-3.1-8B-Sugarquill-v9-apollo"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-unsloth-bnb-4bit",
    max_seq_length=1024 * 8,
    device_map="auto",
    load_in_4bit=True,
    float8_kv_cache=True,
)

print(f"Model loaded. dtype = {model.dtype}.")

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
)

sugarquill = load_dataset("Nelathan/synthetic-sugar-quill", split="train")
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
    desc="Formatting Sugarquill",
)

ds_split = sugarquill.train_test_split(0.02, seed=42)
train_dataset = ds_split["train"]
eval_dataset = ds_split["test"]
print(f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} test samples.")

learning_rate = 2e-4
args = SFTConfig(
    output_dir="outputs/" + project,
    report_to="wandb",
    num_train_epochs=2,
    learning_rate=learning_rate,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    batch_eval_metrics=True,
    # optim="ademamix_8bit",
    optim="apollo_adamw",
    optim_target_modules=[r".*.self_attn.*", r".*.mlp.*"],
    lr_scheduler_type="linear",
    max_grad_norm=0.5,
    warmup_steps=100,
    # weight_decay=0.01,
    logging_steps=1,
    eval_strategy="steps",
    do_eval=True,
    eval_steps=100,
    save_total_limit=3,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset.shuffle(seed=42),
    eval_dataset=eval_dataset,
    args=args,
)


trainer = train_on_responses_only(trainer, "# About me\n\n", "Lets write a story.\n\n")


gc.collect()
torch.cuda.empty_cache()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project="Llama-3.1-8B-Sugarquill", entity="pink-marker", save_code=True)

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

model.save_pretrained(f"outputs/{project}")
tokenizer.save_pretrained(f"outputs/{project}")
# Online saving
model.push_to_hub(f"Nelathan/{project}")
tokenizer.push_to_hub(f"Nelathan/{project}")

# model.save_pretrained_gguf(
#     f"outputs/{project}/q4_k_m", tokenizer, quantization_method="q4_k_m"
# )
print(f"Model saved to output/{project}.")

FastModel.for_inference(model)

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

tests = [
    "Lets write a story.\n\n",
    test_story_profile + "\n\nLets write a story.\n\n",
    test_story_profile
    + "\n\n<thinking>"
    + test_story_notes
    + "</thinking>\n\n# The Alabaster Echo\n\n",
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
