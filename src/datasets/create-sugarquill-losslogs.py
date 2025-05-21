import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

records = []
SEQ_LEN = 1024 * 8

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-pt", device_map="auto", torch_dtype="auto"
)
model.eval()

ds = load_dataset("Nelathan/synthetic-sugar-quill", split="train").shuffle(42)

for idx, row in enumerate(tqdm(ds, desc="calculting losses")):
    enc = el = tokenizer(
        row["text"],
        return_tensors="pt",
        truncation=True,
        max_length=SEQ_LEN,
        add_special_tokens=True,
    ).to("cuda")
    id = row["id"]
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_length = attention_mask.sum().item()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        loss = outputs.loss.item()

    perplexity = np.exp(loss) if not np.isnan(loss) else float("nan")

    records.append(
        {
            "id": row["id"],
            "loss": loss,
            "perplexity": perplexity,
            "token_length": token_length,
        }
    )

    if idx % 100 == 0:
        print(
            f"\r[{idx}] id={id} loss={loss:.4f} ppl={perplexity:.2f} tok={token_length}"
        )

    del input_ids, attention_mask, outputs

df = pd.DataFrame(records)
df["outlier_score"] = (df["loss"] - df["loss"].mean()) / df["loss"].std()
df.to_parquet("row_metrics_final.parquet")
print("All done. Final metrics saved.")
