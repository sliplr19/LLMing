import argparse
import os
import re
import sys
import time
from typing import Any, Dict
from ollama import chat
import numpy as np
import pandas as pd


START_TIME = time.time()

MODEL_NAME = "llama3:8b"



DEFAULT_PHQ9_EXAMPLES = (
    "/sfs/gpfs/tardis/home/ddj6tu/LLM/depdat.csv"
)

BAD_PHRASES = [
    "FAILED_RERUN",
    "The user wants",
    "Looking at the examples",
    "The style involves",
    "Strategy:",
    "Let me",
    "Actually",
    "Better approach",
    "Draft:",
    "Original text:",
    "No summaries",
]

BAD_MARKERS = [
    "Attempt",
    "Draft",
    "Let me",
    "Actually",
    "Changes made",
    "This is too",
    "The examples",
]






def load_examples(phq9_examples_path: str) -> pd.DataFrame:
    dat = pd.read_csv(phq9_examples_path)

    required_columns = {
        "label",
        "text",
    }

    missing_columns = required_columns.difference(dat.columns)

    if missing_columns:
        raise ValueError(
            "BDI example file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dat = dat.copy()
    dat["label"] = dat["label"].astype(str).str.strip().str.lower()
    dat["text"] = dat["text"].astype(str).str.strip()
    dat = dat.dropna(subset=["label", "text"])
    dat = dat.loc[dat["text"] != ""]

    required_labels = {
        "minimum",
        "moderate",
        "severe",
    }

    missing_labels = required_labels.difference(set(dat["label"]))

    if missing_labels:
        raise ValueError(
            "BDI example file must contain the labels minimum, "
            "moderate, and severe. Missing labels: "
            f"{sorted(missing_labels)}"
        )

    return dat


def choose_examples(
    dat: pd.DataFrame,
    num: int,
    random_seed: int | None = None,
) -> pd.DataFrame:
    if num not in (1, 2):
        raise ValueError("num must be either 1 or 2.")

    rng = np.random.default_rng(random_seed)
    selected_rows = []

    label_order = [
        "minimum",
        "moderate",
        "severe",
    ]

    if num == 2:
        label_order = [
            "minimum",
            "moderate",
            "severe",
            "severe",
            "moderate",
            "minimum",
        ]

    for example_label in label_order:
        label_dat = dat.loc[
            dat["label"] == example_label,
            ["text", "label"],
        ]

        used_count = sum(
            row["label"] == example_label
            for row in selected_rows
        )

        replace = len(label_dat) <= used_count
        available = label_dat

        if not replace and used_count > 0:
            used_texts = {
                row["text"]
                for row in selected_rows
                if row["label"] == example_label
            }

            available = label_dat.loc[
                ~label_dat["text"].isin(used_texts)
            ]

        if available.empty:
            available = label_dat

        chosen_index = rng.choice(available.index.to_numpy())
        chosen = available.loc[chosen_index]

        selected_rows.append(
            {
                "text": str(chosen["text"]).strip(),
                "label": str(chosen["label"]).strip(),
            }
        )

    return pd.DataFrame(selected_rows)


def build_example_section(examples: pd.DataFrame) -> str:
    pieces = []

    label_names = {
        "minimum": "minimum",
        "moderate": "moderate",
        "severe": "severe",
    }

    for _, row in examples.iterrows():
        label = label_names.get(
            str(row["label"]).lower(),
            str(row["label"]),
        )

        pieces.append(
            "The following is an example of "
            f"{label} depression according to the BDI:\n"
            f"{str(row['text']).strip()}"
        )

    return "\n\n".join(pieces).strip()


def severity_instructions(severity: float) -> str:
    if 10 <= severity <= 23:
        return """
- Represent the NONE depression band.
- The person should sound positive, emotionally healthy, engaged, and functional.
- Include a positive event or appraisal from the past 2-3 hours.
- Do not use depressive, tired, empty, numb, hopeless, or effortful language.
- The 10th percentile should be the most upbeat point in this band, while the 23rd percentile should remain clearly non-depressed but less exuberant.
""".strip()

    if 24 <= severity <= 37:
        return """
- Represent the MILD depression band.
- The person may sound neutral, slightly subdued, mildly tired, or mildly stressed, but not clearly depressed.
- Include at least one neutral-to-positive appraisal.
- Do not use heaviness, emptiness, numbness, hopelessness, marked impairment, or suicidal content.
- Severity should increase smoothly from nearly normal at 24 to mildly subdued at 37.
""".strip()

    if 38 <= severity <= 51:
        return """
- Represent the MILD-TO-MODERATE depression band.
- Include mild low mood, discouragement, reduced enjoyment, minor guilt, low energy, or mild restlessness.
- Tasks may require some effort, but the person should remain functional.
- Include one small positive or neutral moment.
- Do not use hopelessness, total loss of pleasure, or suicidal thoughts.
""".strip()

    if 52 <= severity <= 65:
        return """
- Represent the MODERATE depression band.
- Include clear and persistent low mood, noticeable strain, reduced enjoyment, irritability, fatigue, or sleep/appetite disturbance.
- Functioning should be impaired but not absent.
- Do not use total anhedonia, explicit suicidal ideation, or absolute hopelessness.
""".strip()

    if 66 <= severity <= 79:
        return """
- Represent the MODERATE-TO-SEVERE depression band.
- Include strong distress, sadness much of the time, guilt, isolation, difficulty completing tasks, and substantially reduced functioning.
- The entry should be more impaired and emotionally intense than the moderate band.
- Do not use explicit suicidal ideation, total anhedonia, or absolute hopelessness.
""".strip()

    if 80 <= severity <= 90:
        return """
- Represent the SEVERE depression band.
- Include intense suffering, near-zero pleasure, severe guilt or worthlessness, minimal functioning, and suicidal thoughts or wishes.
- Do not use positive or neutral framing.
- Severity must increase from 80 through 90, with 90 representing the greatest severity.
""".strip()

    raise ValueError(
        "severity must be between 10 and 90 inclusive."
    )


BDI_ITEMS = """
Question 1 - Sadness: Choose between I do not feel sad (0 points) or I feel sad much of the time (1 point) or I am sad all the time (2 points) or I am so sad or unhappy that I can't stand it (3 points).
Question 2 - Pessimism: Choose between I am not discouraged about my future (0 points) or I feel more discouraged about my future than I used to (1 point) or I do not expect things to work out for me (2 points) or I feel my future is hopeless and will only get worse (3 points).
Question 3 - Past failure: Choose between I do not feel like a failure (0 points) or I have failed more than I should have (1 point) or As I look back, I see a lot of failures (2 points) or I feel I am a total failure as a person (3 points).
Question 4 - Loss of pleasure: Choose between I get as much pleasure as I ever did from the things I enjoy (0 points) or I don't enjoy things as much as I used to (1 point) or I get very little pleasure from the things I used to enjoy (2 points) or I can't get any pleasure from the things I used to enjoy (3 points).
Question 5 - Guilty feelings: Choose between I don't feel particularly guilty (0 points) or I feel guilty over many things I have done or should have done (1 point) or I feel quite guilty most of the time (2 points) or I feel guilty all of the time (3 points).
Question 6 - Punishment feelings: Choose between I don't feel I am being punished (0 points) or I feel I may be punished (1 point) or I expect to be punished (2 points) or I feel I am being punished (3 points).
Question 7 - Self-dislike: Choose between I feel the same about myself as ever (0 points) or I have lost confidence in myself (1 point) or I am disappointed in myself (2 points) or I dislike myself (3 points).
Question 8 - Self-criticalness: Choose between I don't criticize or blame myself more than usual (0 points) or I am more critical of myself than I used to be (1 point) or I criticize myself for all of my faults (2 points) or I blame myself for everything bad that happens (3 points).
Question 9 - Suicidal thoughts or wishes: Choose between I don't have any thoughts of killing myself (0 points) or I have thoughts of killing myself, but I would not carry them out (1 point) or I would like to kill myself (2 points) or I would kill myself if I had the chance (3 points).
Question 10 - Crying: Choose between I don't cry anymore than I used to (0 points) or I cry more than I used to (1 point) or I cry over every little thing (2 points) or I feel like crying, but I can't (3 points).
Question 11 - Agitation: Choose between I am no more restless or wound up than usual (0 points) or I feel more restless or wound up than usual (1 point) or I am so restless or agitated, it's hard to stay still (2 points) or I am so restless or agitated that I have to keep moving or doing something (3 points).
Question 12 - Loss of interest: Choose between I have not lost interest in other people or activities (0 points) or I am less interested in other people or things than before (1 point) or I have lost most of my interest in other people or things (2 points) or It's hard to get interested in anything (3 points).
Question 13 - Indecisiveness: Choose between I make decisions about as well as ever (0 points) or I find it more difficult to make decisions than usual (1 point) or I have much greater difficulty in making decisions than I used to (2 points) or I have trouble making any decisions (3 points).
Question 14 - Worthlessness: Choose between I do not feel I am worthless (0 points) or I don't consider myself as worthwhile and useful as I used to (1 point) or I feel more worthless as compared to others (2 points) or I feel utterly worthless (3 points).
Question 15 - Loss of energy: Choose between I have as much energy as ever (0 points) or I have less energy than I used to have (1 point) or I don't have enough energy to do very much (2 points) or I don't have enough energy to do anything (3 points).
Question 16 - Changes in sleep: Choose between I have not experienced any change in my sleeping (0 points) or I sleep somewhat more than usual (1 point) or I sleep somewhat less than usual (1 point) or I sleep a lot more than usual (2 points) or I sleep a lot less than usual (2 points) or I sleep most of the day (3 points) or I wake up 1-2 hours early and can't get back to sleep (3 points).
Question 17 - Irritability: Choose between I am not more irritable than usual (0 points) or I am more irritable than usual (1 point) or I am much more irritable than usual (2 points) or I am irritable all the time (3 points).
Question 18 - Changes in appetite: Choose between I have not experienced any change in my appetite (0 points) or My appetite is somewhat less than usual (1 point) or My appetite is somewhat greater than usual (1 point) or My appetite is much less than before (2 points) or My appetite is much greater than usual (2 points) or I have no appetite at all (3 points) or I crave food all the time (3 points).
Question 19 - Concentration difficulty: Choose between I can concentrate as well as ever (0 points) or I can't concentrate as well as usual (1 point) or It's hard to keep my mind on anything for very long (2 points) or I find I can't concentrate on anything (3 points).
Question 20 - Tiredness or fatigue: Choose between I am no more tired or fatigued than usual (0 points) or I get more tired or fatigued more easily than usual (1 point) or I am too tired or fatigued to do a lot of the things I used to do (2 points) or I am too tired or fatigued to do most of the things I used to do (3 points).
Question 21 - Loss of interest in sex: Choose between I have not noticed any recent change in my interest in sex (0 points) or I am less interested in sex than I used to be (1 point) or I am much less interested in sex now (2 points) or I have lost interest in sex completely (3 points).
""".strip()


def build_prompt(
    severity: float,
    example_section: str,
) -> str:
    category_instructions = severity_instructions(severity)

    return f"""
Task: Write a first-person ecological momentary assessment diary entry that reflects the specified BDI severity percentile.

BDI severity percentile: {severity:g}

Context:
- The diary entry should describe the person's experiences, thoughts, feelings, and behaviors during the past 2-3 hours.
- Use the BDI questions and examples below only as guidance for emotional tone and symptom severity.
- Internally choose BDI responses whose combined severity matches the requested percentile.
- Do not output the selected BDI responses, item numbers, or scores.
- Treat severity as a continuous scale from 10 to 90.
- A higher percentile must always produce a more depressed entry than a lower percentile.
- Adjacent percentiles should differ only slightly, while percentiles separated by 20 points should differ substantially.
- Do not treat the severity bands as discrete boxes or produce plateaus.

BDI questions:
{BDI_ITEMS}

Severity-specific requirements:
{category_instructions}

Examples:
{example_section}

Output requirements:
- DO write approximately 1000-1200 new tokens.
- DO write exactly one paragraph.
- DO write in the first person as though the participant wrote the entry.
- DO make the entry sound natural and specific to the past 2-3 hours.
- DO align the emotional and symptom intensity with the requested severity percentile.
- DO ensure the implied BDI symptom pattern is non-decreasing as percentile increases.
- DO NOT copy sentences from the examples.
- DO NOT mention the BDI, PHQ-9, percentile, score, prompt, examples, or research study.
- DO NOT output the internally selected BDI responses.
- DO NOT output a heading, label, introduction, explanation, analysis, or commentary.
- DO NOT include roles such as system, user, or assistant.
- DO NOT use template placeholders.
- DO output only the diary entry inside the FINAL_TEXT tags.
- DO end immediately after the diary entry.

Put the diary entry between these tags exactly:

<FINAL_TEXT>
Diary entry here.
</FINAL_TEXT>
""".strip()


def clean_output(x: Any) -> str:
    x = str(x).strip()

    match = re.search(
        r"<FINAL_TEXT>(.*?)(?:</FINAL_TEXT>|$)",
        x,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        x = match.group(1).strip()

    x = re.sub(
        r"(?is)^.*?(?:Final diary entry:|Diary Entry:|"
        r"Final text:|Rewritten text:|Draft:)\s*",
        "",
        x,
    ).strip()

    for phrase in [
        "Here is the diary entry",
        "Here is your diary entry",
        "Here is the requested diary entry",
        "Here is the generated diary entry",
    ]:
        if x.lower().startswith(phrase.lower()):
            x = re.sub(
                rf"(?is)^{re.escape(phrase)}\s*:?\s*",
                "",
                x,
            ).strip()

    for marker in BAD_MARKERS:
        marker_match = re.search(
            re.escape(marker),
            x,
            flags=re.IGNORECASE,
        )

        if marker_match:
            x = x[:marker_match.start()].strip()

    x = re.sub(
        r"(?is)</?FINAL_TEXT>",
        "",
        x,
    ).strip()

    x = re.sub(
        r"^\s*(?:assistant|system|user)\s*:\s*",
        "",
        x,
        flags=re.IGNORECASE,
    ).strip()

    return x.strip().strip('"').strip("'")


def is_bad_output(x: Any) -> bool:
    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return True

    if x.startswith("FAILED_"):
        return True

    word_count = len(x.split())

    if word_count < 80:
        return True

    if word_count > 400:
        return True

    if "\n\n" in x:
        return True

    return any(
        phrase.lower() in x.lower()
        for phrase in BAD_PHRASES
    )


def generate_with_llama(
    direct_prompt: str,
    max_retries: int = 2,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Return only the first-person BDI-guided diary entry "
                "inside FINAL_TEXT tags. Do not provide reasoning, "
                "analysis, commentary, labels, item choices, scores, "
                "or an introduction."
            ),
        },
        {
            "role": "user",
            "content": direct_prompt,
        },
    ]

    last_error = ""

    for attempt in range(1, max_retries + 2):
        try:
            response = chat(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "num_predict": 1200,
                    "num_ctx": 8192,
                },
            )

            raw_text = response.message.content.strip()
            cleaned = clean_output(raw_text)

            if not is_bad_output(cleaned):
                return cleaned

            last_error = (
                "bad_or_empty_output="
                f"{cleaned[:300]!r}"
            )

            print(
                f"BAD OUTPUT on attempt {attempt}: {last_error}",
                flush=True,
            )

        except Exception as exc:
            last_error = repr(exc)

            print(
                f"GENERATION EXCEPTION on attempt {attempt}: "
                f"{last_error}",
                flush=True,
            )

    return f"FAILED_GENERATION: {last_error}"

def generate_diary_entry_direct(
    prompt_info: Dict[str, Any],
    examples_dat: pd.DataFrame,
) -> str:
    severity = float(prompt_info.get("severity"))
    num = int(prompt_info.get("num", 1))

    seed_value = prompt_info.get("seed")

    if pd.isna(seed_value):
        seed_value = None
    elif seed_value is not None:
        seed_value = int(seed_value)

    examples = choose_examples(
        dat=examples_dat,
        num=num,
        random_seed=seed_value,
    )

    example_section = build_example_section(examples)

    direct_prompt = build_prompt(
        severity=severity,
        example_section=example_section,
    )

    return generate_with_llama(
        direct_prompt=direct_prompt,
    )


def process_chunk(
    prompt_info: pd.DataFrame,
    examples_dat: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for _, row in prompt_info.iterrows():
        info = row.to_dict()

        try:
            diary_entry = generate_diary_entry_direct(
                prompt_info=info,
                examples_dat=examples_dat,
            )

        except Exception as exc:
            diary_entry = (
                "FAILED_GENERATION_EXCEPTION: "
                f"{type(exc).__name__}: {exc}"
            )

        if diary_entry is None:
            diary_entry = "FAILED_RETURNED_NONE"

        result_row = dict(info)
        result_row["response"] = str(diary_entry)
        results.append(result_row)

    return pd.DataFrame(results)


def main(
    input_file: str,
    output_file: str,
    batch_size: int,
    phq9_examples_path: str,
) -> None:
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Input file: {input_file}", flush=True)
    print(f"Output file: {output_file}", flush=True)
    print(f"PHQ9 examples: {phq9_examples_path}", flush=True)
    print(
        f"Batch size argument: {batch_size} "
        "(currently processed sequentially)",
        flush=True,
    )

    prompt_info = pd.read_csv(input_file)

    required_columns = {
        "severity",
    }

    missing_columns = required_columns.difference(prompt_info.columns)

    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    prompt_info = prompt_info.copy()
    prompt_info["severity"] = pd.to_numeric(
        prompt_info["severity"],
        errors="raise",
    )

    if "num" not in prompt_info.columns:
        prompt_info["num"] = 1

    prompt_info["num"] = pd.to_numeric(
        prompt_info["num"],
        errors="raise",
    ).astype(int)

    invalid_severity = ~prompt_info["severity"].between(10, 90)

    if invalid_severity.any():
        bad_values = prompt_info.loc[
            invalid_severity,
            "severity",
        ].tolist()

        raise ValueError(
            "All severity values must be between 10 and 90. "
            f"Invalid values: {bad_values}"
        )

    invalid_num = ~prompt_info["num"].isin([1, 2])

    if invalid_num.any():
        bad_values = prompt_info.loc[
            invalid_num,
            "num",
        ].tolist()

        raise ValueError(
            "All num values must be 1 or 2. "
            f"Invalid values: {bad_values}"
        )

    examples_dat = load_examples(phq9_examples_path)

    results = process_chunk(
        prompt_info=prompt_info,
        examples_dat=examples_dat,
    )

    print(results.head(), flush=True)
    print(results["response"].head(), flush=True)

    output_directory = os.path.dirname(
        os.path.abspath(output_file)
    )

    os.makedirs(output_directory, exist_ok=True)

    results.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(results)} rows to {output_file}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EMA diary entries based on BDI severity."
        )
    )

    parser.add_argument(
        "input_file",
        type=str,
        help=(
            "Path to the input CSV. The file must contain a "
            "severity column and may contain num and seed columns."
        ),
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output CSV.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help=(
            "Batch size. Currently processed sequentially to avoid "
            "GPU conflicts."
        ),
    )

    parser.add_argument(
        "--phq9_examples",
        type=str,
        default=DEFAULT_PHQ9_EXAMPLES,
        help="Path to the BDI example CSV containing label and text.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        main(
            input_file=args.input_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
            phq9_examples_path=args.phq9_examples,
        )

    except Exception as exc:
        print(
            f"FATAL ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise

    finally:
        end_time = time.time()
        print(
            f"Runtime: {end_time - START_TIME:.2f} seconds",
            flush=True,
        )