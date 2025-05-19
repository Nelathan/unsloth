from calendar import c
import stat
from idna import encode
from openai import batches
from sympy import N, rem
from unsloth import FastLanguageModel
import re
from numpy import add
from regex import D, F, T
import torch
import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from trl import SFTConfig
from unsloth.chat_templates import train_on_responses_only
import gc
import wandb
import random
from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F

project = "flatpack"
max_seq_length = 1024 * 8

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
)
print(f"Model loaded. dtype = {model.dtype}.")

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
)

print("Model patched successfully.")

# jondurbin / airoboros - 3.2
airoboros = load_dataset("jondurbin/airoboros-3.2", split="train")
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

airoboros = standardize_sharegpt(airoboros)
airoboros = airoboros.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"], tokenize=False, add_generation_prompt=False
        )
    },
    remove_columns=airoboros.column_names,
    batched=True,
    desc="Formatting airoboros dataset",
)

split = airoboros.train_test_split(test_size=0.04)
train_dataset = split["train"]
test_dataset = split["test"]

print(f"Dataset split: {len(train_dataset)} train, {len(test_dataset)} test samples.")


def group_batches_for_flatpack(samples, max_seq_len=1024 * 8, bin_size=256):
    """Group samples into batches with similar aggregate lengths for training limited by max_seq_len.

    Args:
        samples (dataset): Dataset of tokenized input samples.
        max_seq_len (int, optional): Maximum sequence length.
        bin_size (int, optional): Size of each bin. Defaults to 256.
    """
    from collections import defaultdict
    import random
    from datasets import Dataset

    bins = defaultdict(list)
    for idx, sample in enumerate(samples):
        sample_len = len(sample["input_ids"])
        bin_index = sample_len // bin_size
        bins[bin_index].append(idx)

    # bin at index 0 has samples with size 0 to bin_size-1
    # bin at index 1 has samples with size bin_size to 2*bin_size-1
    # bin at index 2 has samples with size 2*bin_size to 3*bin_size-1

    used_samples = set()
    batches = []

    # assign all samples to a batch
    while len(used_samples) < len(samples):
        batch = []
        budget = None  # assign a single sample even if it is larger than the budget
        # first pick the largest bin, update the token budget
        # then estimate which bin has samples that fit in the remaining budget
        for (
            bin_index,
            bin_samples,
        ) in (
            bins.items().__reversed__()
        ):  # this ignores uninitialized bins and starts from the largest bin
            # if the bin is empty, skip it
            if len(bin_samples) == 0:
                continue

            if budget is None:
                # if budget is None, assign the first sample
                sample_id = bin_samples.pop()
                sample = samples[sample_id]
                len_sample = len(sample["input_ids"])
                budget = max_seq_len - len_sample
                batch.append(sample)

            # add samples from the bin to the batch while the budget allows
            while budget >= (bin_size * (bin_index + 1)):
                sample_id = bin_samples.pop()
                sample = samples[sample_id]
                len_sample = len(sample["input_ids"])
                budget -= len_sample
                batch.append(sample)

        pass
        used_samples.update([sample["id"] for sample in batch])
        batches.append(batch)

    # now all samples are assigned to a batch
    # normalize the batch size by padding with empty samples
    max_batch_size = max(len(batch) for batch in batches)
    pseudo = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
    }

    for i, batch in enumerate(batches):
        while len(batch) < max_batch_size:
            batches[i].append(pseudo)

    pass
    # now all batches have the same size
    # shuffle them and return them as a dataset, also return the max batch size
    random.shuffle(batches)
    dataset = Dataset.from_list(batches)
    return dataset, max_batch_size


from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DataCollatorForFlatpack:
    return_position_ids: bool = True
    separator_id: int = -100
    begin_training_tokens: List[int] = None
    end_training_tokens: List[int] = None

    @staticmethod
    def _find_sublist(lst, sub, start=0):
        """Find the first occurrence of a sublist in a list."""
        sub_len = len(sub)
        for i in range(start, len(lst) - sub_len + 1):
            if lst[i : i + sub_len] == sub:
                return i
        return -1

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        input_ids = []
        attention_masks = []
        position_ids = []
        labels = []
        for feature in features:
            ids = feature["input_ids"]
            labels = feature["labels"]
            start = 0
            # find the first occurrence of the begin training tokens, mask by setting labels to -100, the start and end tokens are not masked
            # 1 if no start found, do not mask
            # 2 if start found, mask from 0 to start
            # 3 if no end found its finished
            # 4 if theres another another start, mask between end to start
            # 5 goto 3

            while True:
                start = self._find_sublist(ids, self.begin_training_tokens, start)
                if start == -1:
                    break
                end = self._find_sublist(ids, self.end_training_tokens, start)
                if end == -1:
                    break
                # mask the tokens between the start and end tokens
                for i in range(start + len(self.begin_training_tokens), end):
                    labels[i] = self.separator_id
                # set the position ids to 0 for the masked tokens
                position_ids += [0] * (end - start - len(self.begin_training_tokens))
                # set the attention masks to 0 for the masked tokens
                attention_masks += [0] * (end - start - len(self.begin_training_tokens))
                # set the input ids to 0 for the masked tokens
                input_ids += [0] * (end - start - len(self.begin_training_tokens))
                # set the start to end + 1
                start = end + 1

        # create the final input tensor
        # concat the batch elements and add a -100 between the input ids
        # ensure the position ids stay consitent per batch element
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_masks = torch.tensor(attention_masks).unsqueeze(0)
        position_ids = torch.tensor(position_ids).unsqueeze(0)
        labels = torch.tensor(labels).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
            "labels": labels,
        }


# tokenize the dataset
train_dataset = train_dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=max_seq_length),
    remove_columns=train_dataset.column_names,
    batched=True,
    desc="Tokenizing train dataset",
)

# group the samples into batches
train_dataset, max_batch_size = group_batches_for_flatpack(
    train_dataset,
    max_seq_length=max_seq_length,
)

args = SFTConfig(
    output_dir="outputs/" + project,
    report_to="wandb",
    num_train_epochs=2,
    learning_rate=5e-5,
    per_device_train_batch_size=max_batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=8,
    batch_eval_metrics=True,
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    max_grad_norm=0.4,
    # warmup_steps=100,
    weight_decay=0.01,
    logging_steps=1,
    eval_strategy="steps",
    do_eval=True,
    eval_steps=200,
    save_total_limit=3,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset.shuffle(seed=42),
    eval_dataset=test_dataset,
    args=args,
    tokenizer=tokenizer,
    peft_config=model.peft_config,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
)

# model.save_pretrained(f"outputs/{project}")
# tokenizer.save_pretrained(f"outputs/{project}")

gc.collect()
torch.cuda.empty_cache()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

wandb.init(project="Llama-3.1-8B-Sugarquill", entity="pink-marker", save_code=True)

trainer_stats = trainer.train(resume_from_checkpoint=False)

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

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a rascal ai assistant."},
    {"role": "user", "content": "Please write about the 5 pillars of prosperity."},
]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

encoded = tokenizer(formatted, return_tensors="pt").to("cuda")

outputs = model.generate(
    **encoded,
    max_new_tokens=1024 * 2,
    use_cache=True,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    repetition_penalty=1.1,
    do_sample=True,
)
outputs = tokenizer.batch_decode(outputs)
for output in outputs:
    clean = output.replace("\\n", "\n").replace("\\t", "\t")
    print(clean)
