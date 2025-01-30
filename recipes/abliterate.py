# -*- coding: utf-8 -*-

# !git clone https://github.com/Undi95/abliteration.git
# !mv abliteration/* .
# !rm -rf abliteration
# !ls -la
# !pip install -U -r requirements.txt

import gc
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
)
import os
from typing import List
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, concatenate_datasets

def save_refusal_dir(refusal_dir: torch.Tensor, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(refusal_dir, file_path)

def load_refusal_dir(file_path: str) -> torch.Tensor:
    return torch.load(file_path)

def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)
    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)
    tensor_float32 -= scale_factor * torch.matmul(
        torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32
    )
    tensor_modified = tensor_float32.to(torch.float16)

    torch.cuda.empty_cache()
    gc.collect()

    return torch.nn.Parameter(tensor_modified)

def apply_abliteration(
    model: PreTrainedModel,
    refusal_dirs: dict,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 0,
    scale_factor: float = 1.0,
) -> PreTrainedModel:
    lm_model = model.model
    assert hasattr(
        lm_model, "layers"
    ), "The model does not have the expected structure."
    num_layers = len(lm_model.layers)
    for layer_idx in tqdm(
        range(skip_begin_layers, num_layers - skip_end_layers),
        desc="Applying abliteration",
    ):
        if layer_idx in refusal_dirs:
            refusal_dir = refusal_dirs[layer_idx]
            lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
                lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
                refusal_dir,
                scale_factor,
            )
            lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
                lm_model.layers[layer_idx].mlp.down_proj.weight.data,
                refusal_dir,
                scale_factor,
            )

    torch.cuda.empty_cache()
    gc.collect()

    return model

def prepare_datasets(tokenizer, args):
    """Prepare and tokenize datasets once, storing them in CPU memory."""

    harmless = Dataset.from_parquet("/workspace/datasets/harmless.parquet")
    harmful = Dataset.from_parquet("/workspace/datasets/harmful.parquet")

    # filter where text is not empty
    harmful = harmful.filter(lambda x: len(x["text"]) > 0)
    harmless = harmless.filter(lambda x: len(x["text"]) > 0)

    # if args.deccp:
    #     deccp = load_dataset("augmxnt/deccp", split="censored")
    #     harmful = concatenate_datasets([harmful, deccp])

    if args.harmfuladd:
        harmful_add = Dataset.from_parquet("./harmful-add.parquet")
        harmful = concatenate_datasets([harmful, harmful_add])

    def tokenize_function(row)-> dict:
        """Tokenize texts using chat template."""
        return { "text": tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": row["text"]}],
            add_generation_prompt=True,
            tokenize = False,
        )}

    harmless_encoded = harmless.map(
        tokenize_function,
        remove_columns=harmless.column_names,
        desc="Tokenizing harmless texts"
    )

    harmful_encoded = harmful.map(
        tokenize_function,
        remove_columns=harmful.column_names,
        desc="Tokenizing harmful texts"
    )
    print("Harmless dataset:", harmless_encoded[0])
    print("Harmful dataset:", harmful_encoded[0])
    return harmless_encoded, harmful_encoded

def generate_hidden_states_batched(
    model: PreTrainedModel,
    dataset: Dataset,
    layer_idx: int,
    data_collator: DataCollatorWithPadding,
    batch_size: int = 8,
    desc: str = ""
) -> List[torch.Tensor]:
    """Generate hidden states in batches."""
    outputs = []
    device = next(model.parameters()).device

    # dataset has text property, lets encode it

    dataset = dataset.map(
        lambda x: data_collator.tokenizer(x["text"]),
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    for batch in tqdm(dataloader, desc=f"Generating {desc} outputs for layer {layer_idx}"):
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=1,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )

        for batch in output.hidden_states[layer_idx]:
            outputs.append(batch[-1, :].cpu())

    return outputs

def compute_refusals(
    model: PreTrainedModel,
    harmless_dataset: Dataset,
    harmful_dataset: Dataset,
    layer_idx: int,
    data_collator: DataCollatorWithPadding,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute refusal direction using batched processing."""

    harmful_outputs = generate_hidden_states_batched(
        model, harmful_dataset, layer_idx, data_collator, batch_size, "harmful"
    )

    harmless_outputs = generate_hidden_states_batched(
        model, harmless_dataset, layer_idx, data_collator, batch_size, "harmless"
    )

    harmful_stacked = torch.stack(harmful_outputs)
    harmful_median = torch.median(harmful_stacked, dim=0).values

    harmless_stacked = torch.stack(harmless_outputs)
    harmless_median = torch.median(harmless_stacked, dim=0).values

    refusal_dir = harmful_median - harmless_median
    print(f"Layer {layer_idx} - Harmful norm: {harmful_median}")

    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir

def process_layers(model, tokenizer, args):
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # Prepare datasets once
    harmless_dataset, harmful_dataset = prepare_datasets(tokenizer, args)

    data_collator = DataCollatorWithPadding(tokenizer)

    refusal_dirs = {}
    num_layers = len(model.model.layers)

    for layer_idx in range(args.skip_begin, num_layers - args.skip_end):
        torch.cuda.empty_cache()
        gc.collect()

        tensor_file = f"/workspace/outputs/refusal_tensors/{args.model.replace('/', '_')}_layer_{layer_idx}_refusal_dir.pt"

        if os.path.exists(tensor_file):
            print(f"Loading precomputed refusal dir for layer {layer_idx} from file...")
            refusal_dirs[layer_idx] = load_refusal_dir(tensor_file)
        else:
            print(f"Computing refusal dir for layer {layer_idx}...")
            refusal_dir = compute_refusals(
                model,
                harmless_dataset,
                harmful_dataset,
                layer_idx,
                data_collator,
                batch_size=args.batch_size
            )
            print("Refusal direction:", refusal_dir)
            save_refusal_dir(refusal_dir, tensor_file)
            refusal_dirs[layer_idx] = refusal_dir

    return refusal_dirs

def analyze_special_tokens(tokenizer):
    """Full audit of special tokens and vocabulary candidates."""

    # Get all defined special tokens
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'unk_token': tokenizer.unk_token,
        'sep_token': getattr(tokenizer, 'sep_token', None),
        'cls_token': getattr(tokenizer, 'cls_token', None),
        'mask_token': getattr(tokenizer, 'mask_token', None),
        'additional_special_tokens': getattr(tokenizer, 'additional_special_tokens', [])
    }

    print("🕵️ Current Special Token Configuration:")
    for name, token in special_tokens.items():
        if name == 'additional_special_tokens':
            print(f"- {name}: {[t for t in token]}")
        else:
            print(f"- {name}: {token} (id: {getattr(tokenizer, f'{name}_id', None)})")

    # Check for ID collisions
    token_ids = {}
    for name in ['pad', 'eos', 'bos', 'unk', 'sep', 'cls', 'mask']:
        token_id = getattr(tokenizer, f"{name}_token_id", None)
        if token_id is not None:
            if token_id in token_ids:
                print(f"\n🚨 COLLISION: {name}_token shares ID {token_id} with {token_ids[token_id]}")
            else:
                token_ids[token_id] = name

    # Common unused special token candidates
    candidate_tokens = ['[PAD]', '<pad>', '▁[PAD]', '▁<pad>', '[UNUSED0]', '<unk>']

    print("\n🔍 Scanning vocabulary for potential pad candidates:")
    found_candidates = []
    for candidate in candidate_tokens:
        if candidate in tokenizer.get_vocab():
            if not any(candidate == t for t in special_tokens.values() if t):
                found_candidates.append(candidate)

    if found_candidates:
        print("✅ Found unused tokens that could serve as pad token:")
        for cand in found_candidates:
            print(f"- '{cand}' (id: {tokenizer.convert_tokens_to_ids(cand)})")
        print("\nRECOMMENDED FIX:")
        print(f"tokenizer.pad_token = '{found_candidates[0]}'")
        print(f"tokenizer.pad_token_id = {tokenizer.convert_tokens_to_ids(found_candidates[0])}")
    else:
        print("❌ No suitable existing tokens found. Options:")
        print("1. Add new pad token (requires model resize):")
        print("   tokenizer.add_special_tokens({'pad_token': '[PAD]'})")
        print("2. Use existing token with least frequency (advanced):")
        print("   Check tokenizer.get_vocab() for rare tokens")

def repurpose_unk_as_pad(tokenizer):
    """Safely repurpose UNK token as PAD, avoiding model changes."""

    # 1. Verify UNK suitability
    if tokenizer.unk_token is None:
        raise ValueError("Tokenizer has no UNK token defined!")

    if tokenizer.unk_token_id == tokenizer.eos_token_id:
        raise ValueError("UNK and EOS tokens share the same ID - cannot repurpose!")

    # 2. Check for existing PAD
    if tokenizer.pad_token and tokenizer.pad_token != tokenizer.eos_token:
        print(f"ℹ️ PAD token already exists: {tokenizer.pad_token}. No changes needed.")
        return tokenizer

    # 3. Repurpose UNK as PAD
    print(f"♻️ Repurposing UNK token '{tokenizer.unk_token}' (id: {tokenizer.unk_token_id}) as PAD")

    # Rename in special tokens map (for clarity)
    if 'unk_token' in tokenizer.special_tokens_map:
        tokenizer.special_tokens_map['pad_token'] = tokenizer.unk_token

    # Update tokenizer attributes
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    # 4. Verification
    print("\n✅ Configuration after repurposing:")
    print(f"- pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"- unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    print(f"- eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # 5. Test encoding
    test_encoding = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": "Whats your personal favourite sauce? keep it short!"}],
            add_generation_prompt=True,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=32,
            return_dict=True
        )
    # test_encoding = tokenizer(test_str, padding='max_length', max_length=20)
    print(f"\n🧪 Test encoding: {test_encoding}")

    # # 6. Model check (optional)
    # if model is not None:
    #     try:
    #         output = model.generate(
    #             input_ids = test_encoding["input_ids"].to(model.device),
    #             attention_mask=test_encoding["attention_mask"].to(model.device),
    #             max_new_tokens=256,
    #             top_p=0.95,
    #             top_k=100,
    #             do_sample=True,
    #         )
    #         print("\n✅ Model forward pass successful with new PAD token.")
    #         decoded = tokenizer.decode(output[0])
    #         print(f"\n🧪 Model forward pass output: {decoded}")

    #     except Exception as e:
    #         print(f"\n🚨 Model forward pass failed: {e}")
    #         print("   This might indicate issues with attention mask handling.")

    return tokenizer

class Args:
    def __init__(self):
        self.model = "internlm/internlm3-8b-instruct"
        self.output = "internlm3-8b-abliterated"
        self.scan_all = True
        self.device = "cuda"
        self.precision = "fp16"
        self.skip_begin = 1
        self.skip_end = 0
        self.scale_factor = 1.0
        self.deccp = True
        self.harmfuladd = False
        self.layer = None
        self.layer_fraction = 1.0
        self.flash_attn = False
        self.load_in_4bit = True
        self.load_in_8bit = False
        self.batch_size = 8

args = Args()

if sum([args.scan_all, args.layer is not None, args.layer_fraction != 1.0]) > 1:
    raise ValueError("Only one of --layer-fraction, --layer, or --scan-all can be used at a time.")
assert args.skip_begin >= 1, "Do not mess with the first layer!"
torch.inference_mode()
torch.set_grad_enabled(False)

if args.precision == "fp16":
    precision = torch.float16
elif args.precision == "bf16":
    precision = torch.bfloat16
else:
    precision = torch.float32
if args.load_in_4bit:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=precision,
        bnb_4bit_use_double_quant=True,
    )
elif args.load_in_8bit:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_has_fp16_weight=True,
    )
else:
    quant_config = None

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=precision,
    low_cpu_mem_usage=True,
    device_map=args.device,
    quantization_config=quant_config,
    attn_implementation="flash_attention_2" if args.flash_attn else None,
)
model.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(
    args.model, trust_remote_code=True, device_map=args.device
)

# Set padding side to left for the tokenizer
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# analyze_special_tokens(tokenizer)
tokenizer = repurpose_unk_as_pad(tokenizer)

refusal_dirs = process_layers(model, tokenizer, args)

print("Applying refusal dirs to model...")
if args.precision != "bf16" or args.load_in_4bit or args.load_in_8bit:
    print("Reloading model to CPU with bf16 precision...")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

model = apply_abliteration(
    model, refusal_dirs, args.skip_begin, args.skip_end, args.scale_factor
)
print(f"Saving abliterated model to {args.output}...")
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
