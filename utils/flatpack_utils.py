import math
import torch
from transformers import default_data_collator
from torch.utils.data import SequentialSampler
from trl import SFTTrainer
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DefaultDataCollator, default_data_collator

used_start = 0
total = 0


def group_batches_for_flatpack(samples, max_seq_length=1024 * 8, bin_size=256):
    """Group samples into batches with similar aggregate lengths for training runs limited by max_seq_length.

    Args:
        samples (dataset): Dataset of tokenized input samples.
        max_seq_length (int, optional): Maximum sequence length.
        bin_size (int, optional): Size of each bin. Defaults to 256.
    """
    from collections import defaultdict
    import random
    from datasets import Dataset
    from datasets import concatenate_datasets
    from tqdm import tqdm

    print(f"Grouping samples into batches with max_seq_length = {max_seq_length}.")
    bins = defaultdict(list)
    for idx, sample in tqdm(
        enumerate(samples), total=len(samples), desc="Binning samples"
    ):
        sample_len = len(sample["input_ids"])
        bin_index = math.ceil(sample_len / bin_size)
        bins[bin_index].append(idx)

    used_samples = set()
    batches = []

    # assign all samples to a batch
    while len(used_samples) < len(samples):
        batch = []
        budget = max_seq_length
        for (
            bin_index,
            bin_samples,
        ) in bins.items().__reversed__():
            while len(bin_samples) > 0 and budget >= (bin_size * (bin_index)):
                sample_id = bin_samples.pop()
                sample = samples[sample_id]
                len_sample = len(sample["input_ids"])
                if len_sample > budget:
                    print(
                        f"Sample {sample_id} exceeds budget ({budget}). Estimated length: {(bin_size * (bin_index))}. Actual length: {len_sample}."
                    )
                    continue
                budget -= len_sample
                batch.append(sample_id)
                used_samples.add(sample_id)

        pass
        if len(batch) == 0:
            break
        batches.append(batch)

    print(f"Grouped into {len(batches)} batches.")
    random.shuffle(batches)
    # now all samples are assigned to a batch
    # normalize the batch size by padding with empty samples (-1)
    max_batch_size = max(len(batch) for batch in batches)

    next_sample_id = len(samples)
    pseudo_samples = []
    # while batch size smaller than max_batch_size, create a pseudo sample with the same structure as the original samples but with empty input_ids
    # add all the pseudo samples to the dataset at once. while creating pseudo samples, let them have a new unique sample_id and update batches with the new sample_id
    for i, batch in enumerate(batches):
        # test if batches are <= max_batch_size in total token length
        batch_len = sum(len(samples[sample_id]["input_ids"]) for sample_id in batch)
        if batch_len > max_seq_length:
            print(
                f"Batch {i} exceeds max_seq_length ({batch_len}). This should not happen."
            )
            raise ValueError("Batch exceeds max_seq_length.")

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
    print(f"Adding {len(pseudo_samples)} pseudo samples to dataset.")

    pseudo_dataset = Dataset.from_list(pseudo_samples)

    print("Pseudo dataset length:", len(pseudo_dataset))
    print("Original dataset length:", len(samples))
    dataset = concatenate_datasets([samples, pseudo_dataset])
    print("Dataset length:", len(dataset))
    # assign each sample its batch_id then sort the dataset by batch_id
    dataset = dataset.map(
        lambda row, row_idx: {"batch_id": sampleid_to_batchid[row_idx]},
        with_indices=True,
    )

    dataset = dataset.sort("batch_id")

    return dataset, max_batch_size


@dataclass
class DataCollatorForFlatpack(DefaultDataCollator):
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

        real_feats = [f for f in features if f["input_ids"][0] != -100]
        packed = {"input_ids": [], "labels": [], "attention_mask": []}
        if self.return_position_ids:
            packed["position_ids"] = []

        # pack features with no cross-example tokens, mask first label of each example
        for f in real_feats:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            lbls = f.get("labels", ids)

            # extend input_ids & attention_mask
            packed["input_ids"] += ids
            packed["attention_mask"] += mask

            # mask first label id per example as -100, keep rest
            new_lbls = [-100] + lbls[1:]
            packed["labels"] += new_lbls

            # restart position_ids at 0 for this feature
            if self.return_position_ids:
                packed["position_ids"] += list(range(len(ids)))

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
    def _get_train_sampler(self):
        if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
            return None
        return SequentialSampler(self.train_dataset)
