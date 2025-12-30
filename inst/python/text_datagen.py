# text_datagen.py

# -------------------------------------------------------------------
# Import
# -------------------------------------------------------------------
import os
import gc
import argparse
import json
import re

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

MODEL = None
TOKENIZER = None


# ---------------------------
# Load prompt config & examples
# ---------------------------
def load_prompt_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required_keys = [
        "scenario",
        "overall_rules",
        "percentile_scaffold",
        "item_rules",
        "items",
        "structure_rules",
        "percentile_specification",
        "band_specification",
        "example_instruction",
        "what_to_write",
        "task_desc",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing keys in config file: {missing}")
    return cfg


def load_examples(example_path: str) -> pd.DataFrame:
    df = pd.read_csv(example_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("examples file must have columns 'text' and 'label'.")
    return df


# ---------------------------
# Model and tokenizer setup
# ---------------------------
def load_model_and_tokenizer(model_name: str):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    config = AutoConfig.from_pretrained(model_name)

    # Optional: rope scaling tweak (harmless if absent)
    if hasattr(config, "rope_scaling"):
        config.rope_scaling = {
            "type": "linear",
            "factor": 2.0,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Decide device and dtype ourselves
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)

    return model, tokenizer


def ensure_model_loaded(model_name: str):
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        MODEL, TOKENIZER = load_model_and_tokenizer(model_name)


# ---------------------------
# Choose examples
# ---------------------------
def choose_examples(example_df: pd.DataFrame,
                    num_examples: int,
                    label_col: str = "label") -> pd.DataFrame:
    """
    General version, analogous to the R function:

      groups <- split(dat, dat[[label_col]])
      sampled <- lapply(groups, function(d) {
        n <- min(num, nrow(d))
        d[sample(seq_len(nrow(d)), n), , drop = FALSE]
      })

    Here we:
      * group by `label_col`
      * sample up to `num_examples` rows per group
      * return a single concatenated DataFrame
    """
    if label_col not in example_df.columns:
        raise ValueError(f"label_col='{label_col}' not found in examples.")

    grouped = example_df.groupby(label_col, dropna=False)

    sampled_frames = []
    num_examples = int(num_examples)

    for _, g in grouped:
        n = min(num_examples, len(g))
        if n > 0:
            sampled_frames.append(g.sample(n=n, replace=False, random_state=None))

    if not sampled_frames:
        raise ValueError("No examples available to sample from.")

    return pd.concat(sampled_frames, axis=0)


# ---------------------------
# Cleaning generated text
# ---------------------------
def clean_diary(text: str) -> str:
    t = text.strip()

    t = re.sub(
        r"Here is the internal mapping:.*?And now the diary entry:\s*",
        "",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )

    t = re.sub(
        r'^\s*"{0,2}\s*(here\s+is|here\'?s)\s+'
        r'(a\s+possible\s+)?(the\s+written\s+)?'
        r'(diary|journal)\s+entry\s*:?\s*',
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r'^\s*"{0,2}\s*here\'?s\s+my\s+attempt\s*:?\s*',
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(r"^.*?:\s*", "", t)

    t = t.strip().strip('"').strip("'").strip()

    t = re.sub(
        r'\(Note:.*$',
        "",
        t,
        flags=re.DOTALL | re.IGNORECASE,
    )
    t = re.sub(
        r'Note:.*$',
        "",
        t,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return t.strip()


# ---------------------------
# Generate diary for a single row
# ---------------------------
def generate_diary_entry_direct(row: dict,
                                examples_df: pd.DataFrame,
                                cfg: dict,
                                model_name: str) -> str:
    ensure_model_loaded(model_name)

    severity = row.get("severity")
    num_examples = int(row.get("num_examples", 2))

    # generalized example sampling
    examples = choose_examples(examples_df, num_examples, label_col="label")

    example_lines = []
    for _, ex in examples.iterrows():
        example_lines.append(
            f"The following is an example of {ex['label']}:\n{ex['text']}\n"
        )
    example_section = "\n".join(example_lines)

    scenario = cfg["scenario"]
    overall_rules = cfg["overall_rules"]
    percentile_scaffold = cfg["percentile_scaffold"]
    item_rules = cfg["item_rules"]
    items = cfg["items"]
    structure_rules = cfg["structure_rules"]
    percentile_specification = cfg["percentile_specification"]
    band_specification = cfg["band_specification"]
    example_instruction = cfg["example_instruction"]
    what_to_write = cfg["what_to_write"]
    task_desc = cfg["task_desc"]

    # generation hyperparameters from config (with defaults)
    target_min_cfg = int(cfg.get("target_min", 90))
    target_max_cfg = int(cfg.get("target_max", 100))
    temperature = float(cfg.get("temperature", 0.4))
    top_p = float(cfg.get("top_p", 0.9))
    repetition_penalty = float(cfg.get("repetition_penalty", 1.1))

    direct_prompt = f"""
SCENARIO
-------------------------------------------------------
{scenario}

OVERALL RULES
------------------------
{overall_rules}

PERCENTILE SCAFFOLD
-----------------------
{percentile_scaffold}

ITEM RULES
---------------
{item_rules}

MEASURE ITEMS
------------------------
{items}

STRUCTURE RULES
-----------------------
{structure_rules}

PERCENTILE SPECIFICATION
-----------------------
{percentile_specification}

{band_specification}

EXAMPLES
--------------
{example_instruction}

{example_section}

WHAT TO WRITE
-------------
The current participant is at the {severity}th percentile of depression severity.
{what_to_write}

Diary Entry:
"""

    messages = [
        {"role": "system", "content": task_desc},
        {"role": "user", "content": direct_prompt},
    ]

    if (
        hasattr(TOKENIZER, "apply_chat_template")
        and getattr(TOKENIZER, "chat_template", None)
    ):
        input_text = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = f"[SYSTEM]\n{task_desc}\n\n[USER]\n{direct_prompt}"


    
    inputs = TOKENIZER(
        input_text,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Use whatever device the model is on
    device = next(MODEL.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]
    ctx = getattr(MODEL.config, "max_position_embeddings", 8192)
    available = max(0, ctx - prompt_len - 8)

    # cap by available context
    target_min = max(0, min(target_min_cfg, available))
    target_max = max(target_min, min(target_max_cfg, available))

    outputs = MODEL.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=target_max,
        min_new_tokens=target_min,
        do_sample=True,
        pad_token_id=TOKENIZER.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    generated_tokens = outputs[0, prompt_len:]
    raw_text = TOKENIZER.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    diary_entry = clean_diary(raw_text)
    return diary_entry


# ---------------------------
# Process a whole CSV
# ---------------------------
def process_prompts(input_file: str,
                    examples_df: pd.DataFrame,
                    cfg: dict,
                    batch_size: int,
                    model_name: str):
    prompt_df = pd.read_csv(input_file)
    if "severity" not in prompt_df.columns:
        raise ValueError("input CSV must contain a 'severity' column.")

    results = []
    for _, row in prompt_df.iterrows():
        info = row.to_dict()
        print("Processing:", info)
        diary_entry = generate_diary_entry_direct(info, examples_df, cfg, model_name)
        results.append({
            "id": info.get("id"),
            "severity": info.get("severity"),
            "response": diary_entry,
        })

    return pd.DataFrame(results)


def main(input_file: str,
         output_file: str,
         batch_size: int,
         example_file: str,
         config_file: str,
         model_name: str):
    cfg = load_prompt_config(config_file)
    examples_df = load_examples(example_file)

    out_df = process_prompts(
        input_file=input_file,
        examples_df=examples_df,
        cfg=cfg,
        batch_size=batch_size,
        model_name=model_name,
    )
    out_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate diary entries based on participant information."
    )
    parser.add_argument("input_file", type=str,
                        help="Path to the input CSV with participant information.")
    parser.add_argument("output_file", type=str,
                        help="Path to save the output CSV with diary entries.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (currently not heavily used).")
    parser.add_argument("--example_file", type=str, required=True,
                        help="Path to examples.csv provided by the caller.")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to prompt_config.json provided by the caller.")
    parser.add_argument("--model_name", type=str,
                        default="sshleifer/tiny-gpt2",
                        help="Model name/path recognized by transformers.")

    args = parser.parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        example_file=args.example_file,
        config_file=args.config_file,
        model_name=args.model_name,
    )
