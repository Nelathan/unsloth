import torch
from transformers import default_data_collator
from torch.utils.data import SequentialSampler
from trl import SFTTrainer
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DefaultDataCollator, default_data_collator
from collections import defaultdict
from datasets import Dataset, concatenate_datasets

used_start = 0
total = 0


def group_batches_for_flatpack(samples, max_seq_length=8192, bin_count=32):
    """
    Groups samples into batches with similar aggregate lengths for efficient packing.

    Args:
        samples (Dataset): Tokenized input samples.
        max_seq_length (int): Maximum sequence length per batch.
        bin_count (int): Number of bins to group samples by length.

    Returns:
        (Dataset, int): Tuple of (batched dataset, max batch size).
    """
    bins = defaultdict(list)
    for idx, sample in enumerate(samples):
        sample_len = len(sample["input_ids"])
        bin_index = min(bin_count - 1, sample_len * bin_count // max_seq_length)
        bins[bin_index].append(idx)

    used_samples = set()
    batches = []

    while len(used_samples) < len(samples):
        batch = []
        budget = max_seq_length
        # Iterate bins from largest to smallest
        for bin_index in reversed(range(bin_count)):
            bin_samples = bins[bin_index]
            while bin_samples and budget >= 0:
                sample_id = bin_samples[-1]
                sample = samples[sample_id]
                len_sample = len(sample["input_ids"])
                if len_sample > budget:
                    break
                bin_samples.pop()
                budget -= len_sample
                batch.append(sample_id)
                used_samples.add(sample_id)
        if not batch:
            break
        batches.append(batch)

    # Pad batches to uniform size with pseudo-samples
    max_batch_size = max(len(batch) for batch in batches)
    pseudo_samples = []
    next_sample_id = len(samples)
    for batch in batches:
        while len(batch) < max_batch_size:
            pseudo_samples.append(
                {
                    "input_ids": [-100],
                    "attention_mask": [0],
                }
            )
            batch.append(next_sample_id)
            next_sample_id += 1

    sampleid_to_batchid = {
        sample_id: i for i, batch in enumerate(batches) for sample_id in batch
    }

    pseudo_dataset = Dataset.from_list(pseudo_samples)
    dataset = concatenate_datasets([samples, pseudo_dataset])

    dataset = dataset.map(
        lambda row, row_idx: {"batch_id": sampleid_to_batchid.get(row_idx, -1)},
        with_indices=True,
    )
    dataset = dataset.sort("batch_id").remove_columns("batch_id")

    return dataset, max_batch_size


@dataclass
class DataCollatorForFlatpack(DefaultDataCollator):
    """
    Packs a batch of features into a single sequence for flat-packing.
    Masks the first label of each example, restarts position_ids per example to prevent cross contamination of attention.
    """

    return_position_ids: bool = True
    separator_id: int = -100

    def __init__(self, *args, return_position_ids=True, separator_id=-100, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None, separator_id=None
    ) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        # Filter out pseudo-samples
        real_feats = [f for f in features if f["input_ids"][0] != -100]
        packed = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        if self.return_position_ids:
            packed["position_ids"] = []

        for f in real_feats:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            lbls = f.get("labels", ids)
            packed["input_ids"].extend(ids)
            packed["attention_mask"].extend(mask)
            # Mask first label per example
            packed["labels"].append(-100)
            packed["labels"].extend(lbls[1:])
            if self.return_position_ids:
                packed["position_ids"].extend(range(len(ids)))

        return default_data_collator([packed], return_tensors)


def vram_report_start(stage: str):
    gpu = torch.cuda.get_device_properties(0)
    used_start = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    total = round(gpu.total_memory / 1024**3, 3)
    print(
        f"[{stage}] GPU={gpu.name}, used={used_start}GB/{total}GB ({used_start/total*100:.1f}%)"
    )


def vram_report_end(trainers_stats):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - used_start, 3)
    used_percentage = round(used_memory / total * 100, 3)
    lora_percentage = round(used_memory_for_lora / total * 100, 3)
    print(f"{trainers_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


class FlatpackTrainer(SFTTrainer):
    """
    Trainer with a deterministic sequential sampler for flat-packed batches.
    """

    def _get_train_sampler(self):
        if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
            return None
        return SequentialSampler(self.train_dataset)
