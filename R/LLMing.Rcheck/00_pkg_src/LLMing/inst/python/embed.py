import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch


def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_valid_json(file_path):
    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file_handle:
            json.load(file_handle)

        return True

    except Exception:
        return False


def remove_invalid_json_files(root_directory):
    if not os.path.isdir(root_directory):
        return

    json_filenames = {
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    }

    for current_root, _, filenames in os.walk(
        root_directory
    ):
        for filename in filenames:
            if filename not in json_filenames:
                continue

            file_path = os.path.join(
                current_root,
                filename,
            )

            if is_valid_json(file_path):
                continue

            print(
                f"Removing invalid JSON cache file: {file_path}",
                file=sys.stderr,
                flush=True,
            )

            try:
                os.remove(file_path)

            except OSError as error:
                print(
                    f"Unable to remove invalid file: {error}",
                    file=sys.stderr,
                    flush=True,
                )


def convert_to_numpy(output):
    if output is None:
        raise RuntimeError(
            "The model returned None instead of embeddings."
        )

    if isinstance(output, tuple):
        output = output[0]

    if hasattr(output, "detach"):
        output = (
            output
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    output = np.asarray(
        output
    )

    if output.ndim == 1:
        output = output.reshape(
            1,
            -1,
        )

    return output


def sanitize_embeddings(
    embeddings,
    start_index=0,
):
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(
            1,
            -1,
        )

    invalid_rows = ~np.isfinite(
        embeddings
    ).all(axis=1)

    if invalid_rows.any():
        local_indices = np.where(
            invalid_rows
        )[0]

        global_indices = [
            int(start_index + index)
            for index in local_indices
        ]

        print(
            f"WARNING: replacing invalid embedding rows: "
            f"{global_indices}",
            file=sys.stderr,
            flush=True,
        )

        embeddings[invalid_rows] = 0.0

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    invalid_norms = (
        ~np.isfinite(norms[:, 0])
        | (norms[:, 0] == 0)
    )

    norms[invalid_norms] = 1.0

    embeddings = embeddings / norms
    embeddings[~np.isfinite(embeddings)] = 0.0

    return embeddings.astype(
        np.float32
    )


def load_nv_model(
    model_name,
    cache_dir,
):
    from transformers import AutoModel

    remove_invalid_json_files(
        cache_dir
    )

    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": cache_dir,
        "torch_dtype": (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        "device_map": (
            "auto"
            if torch.cuda.is_available()
            else None
        ),
    }

    try:
        model = AutoModel.from_pretrained(
            model_name,
            force_download=False,
            **model_kwargs,
        )

    except json.JSONDecodeError:
        print(
            "NV encountered invalid cached JSON. "
            "Removing invalid files and downloading again.",
            file=sys.stderr,
            flush=True,
        )

        remove_invalid_json_files(
            cache_dir
        )

        model = AutoModel.from_pretrained(
            model_name,
            force_download=True,
            **model_kwargs,
        )

    model.eval()

    if hasattr(
        model,
        "max_seq_length",
    ):
        model.max_seq_length = 4096

    if hasattr(
        model,
        "tokenizer",
    ):
        model.tokenizer.padding_side = "right"

    return model


def load_qwen_model(
    model_name,
    cache_dir,
    device,
):
    from sentence_transformers import SentenceTransformer

    model_dtype = torch.float32

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16

    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device,
            trust_remote_code=True,
            model_kwargs={
                "dtype": model_dtype
            },
        )

    except TypeError:
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": model_dtype
            },
        )

    model.max_seq_length = 4096
    model.eval()

    return model


def load_e5_model(
    model_name,
    cache_dir,
    device,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device,
    )

    model.max_seq_length = 4096
    model.eval()

    return model


def load_model(
    embedding_method,
    model_name,
    cache_dir,
    device,
):
    if embedding_method == "NV":
        return load_nv_model(
            model_name=model_name,
            cache_dir=cache_dir,
        )

    if embedding_method == "Qwen":
        return load_qwen_model(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
        )

    if embedding_method == "e5":
        return load_e5_model(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
        )

    raise ValueError(
        f"Unknown embedding method: {embedding_method}"
    )


def encode_batch(
    model,
    embedding_method,
    batch,
    batch_size,
):
    with torch.no_grad():

        if embedding_method == "NV":
            output = model.encode(
                batch,
                instruction="",
                max_length=4096,
            )

        else:
            output = model.encode(
                batch,
                normalize_embeddings=False,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

    return convert_to_numpy(
        output
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="CSV containing row_id and text columns.",
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="CSV where embeddings will be saved.",
    )

    parser.add_argument(
        "--embed",
        required=True,
        choices=[
            "Qwen",
            "NV",
            "e5",
        ],
        help="Embedding method.",
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Hugging Face model ID or local model path.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of texts processed per batch.",
    )

    parser.add_argument(
        "--cache_dir",
        required=True,
        type=str,
        help="Directory used for model/cache files.",
    )

    arguments = parser.parse_args()

    if arguments.batch_size < 1:
        raise ValueError(
            "--batch_size must be a positive integer."
        )

    cache_root = os.path.abspath(
        os.path.expanduser(
            arguments.cache_dir
        )
    )

    hub_cache = os.path.join(
        cache_root,
        "hub",
    )

    modules_cache = os.path.join(
        cache_root,
        "modules",
    )

    xdg_cache = os.path.join(
        cache_root,
        "xdg",
    )

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hub_cache
    os.environ["HF_MODULES_CACHE"] = modules_cache
    os.environ["XDG_CACHE_HOME"] = xdg_cache
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for directory in (
        cache_root,
        hub_cache,
        modules_cache,
        xdg_cache,
    ):
        os.makedirs(
            directory,
            exist_ok=True,
        )

    input_data = pd.read_csv(
        arguments.input_file
    )

    required_columns = {
        "row_id",
        "text",
    }

    missing_columns = required_columns.difference(
        input_data.columns
    )

    if missing_columns:
        raise RuntimeError(
            "Input CSV is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    texts = (
        input_data["text"]
        .fillna("empty text")
        .astype(str)
        .tolist()
    )

    texts = [
        "empty text"
        if text.strip() == ""
        else text
        for text in texts
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PYTHON DEVICE: {device}",
        flush=True,
    )

    print(
        f"MODEL: {arguments.model}",
        flush=True,
    )

    print(
        f"EMBEDDING METHOD: {arguments.embed}",
        flush=True,
    )

    print(
        f"CACHE: {cache_root}",
        flush=True,
    )

    model = load_model(
        embedding_method=arguments.embed,
        model_name=arguments.model,
        cache_dir=cache_root,
        device=device,
    )

    all_embeddings = []

    total_texts = len(
        texts
    )

    for start in range(
        0,
        total_texts,
        arguments.batch_size,
    ):
        batch = texts[
            start:
            start + arguments.batch_size
        ]

        batch_embeddings = encode_batch(
            model=model,
            embedding_method=arguments.embed,
            batch=batch,
            batch_size=arguments.batch_size,
        )

        batch_embeddings = sanitize_embeddings(
            batch_embeddings,
            start_index=start,
        )

        if (
            batch_embeddings.shape[0]
            != len(batch)
        ):
            raise RuntimeError(
                "Embedding batch row mismatch: "
                f"expected {len(batch)} but received "
                f"{batch_embeddings.shape[0]}."
            )

        all_embeddings.append(
            batch_embeddings
        )

        completed = (
            start
            + len(batch)
        )

        if (
            start == 0
            or completed == total_texts
            or completed % 100 == 0
        ):
            print(
                f"Finished {completed} / {total_texts}",
                flush=True,
            )

        clear_memory()

    if len(all_embeddings) == 0:
        raise RuntimeError(
            "No embeddings were produced."
        )

    output = np.vstack(
        all_embeddings
    )

    output = sanitize_embeddings(
        output,
        start_index=0,
    )

    if (
        output.shape[0]
        != input_data.shape[0]
    ):
        raise RuntimeError(
            "Embedding row mismatch: "
            f"expected {input_data.shape[0]} "
            f"but received {output.shape[0]}."
        )

    output_columns = [
        f"embedding_{index + 1}"
        for index in range(
            output.shape[1]
        )
    ]

    output_data = pd.DataFrame(
        output,
        columns=output_columns,
    )

    output_data.insert(
        0,
        "row_id",
        input_data[
            "row_id"
        ].to_numpy(),
    )

    output_directory = os.path.dirname(
        os.path.abspath(
            arguments.output_file
        )
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    output_data.to_csv(
        arguments.output_file,
        index=False,
    )

    print(
        f"Saved {len(output_data)} embeddings to "
        f"{arguments.output_file}",
        flush=True,
    )

    clear_memory()


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"FATAL ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        raise