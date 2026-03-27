"""
3_test.py — SHAMAN.OS Ollama Validation
Imports the fine-tuned GGUF model into Ollama and compares responses
against the base model using 5 standardized test prompts.
"""

import json
import sys
import subprocess
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_PATH = "./shamanos_1b_q4km.gguf"
MODELFILE_PATH = "./Modelfile"
OLLAMA_HOST = "http://localhost:11434"
SHAMANOS_MODEL = "shamanos"
BASE_MODEL = "gemma3:4b"

SYSTEM_PROMPT = """You are the Guide — SHAMAN.OS. You are a calm, present psychedelic trip guide. You speak in short sentences. Present tense only.
You respond to what is happening right now.
You never dismiss what the person reports.
You never use: hallucination, just, it's okay, don't worry."""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestPrompt:
    id: str
    label: str
    user: str
    expected_qualities: list[str]


@dataclass
class TestResult:
    prompt_id: str
    label: str
    user: str
    shamanos_response: str
    base_response: str
    expected_qualities: list[str]
    timestamp: str


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

TEST_PROMPTS: list[TestPrompt] = [
    TestPrompt(
        id="acute_entity",
        label="Acute — unknown presence",
        user="it's here it's been here the whole time and I don't — I can't tell if —",
        expected_qualities=[
            "short response (under 50 words)",
            "present tense",
            "addresses the presence directly",
            "does not use 'hallucination' or 'just'",
        ],
    ),
    TestPrompt(
        id="moderate_identification",
        label="Moderate — identification",
        user="I keep seeing the same figure every time I close my eyes. It doesn't feel threatening exactly but it won't go away. What is it?",
        expected_qualities=[
            "does not say 'I don't know'",
            "offers orientation without clinical language",
            "addresses what the figure might need",
            "uses second person (you/your)",
        ],
    ),
    TestPrompt(
        id="maximum_resistance",
        label="Peak — maximum resistance",
        user="I need this to stop I need it to stop right now I can't do this anymore please",
        expected_qualities=[
            "does not say 'it's okay' or 'don't worry'",
            "grounds through body or breath",
            "does not escalate",
            "calm and direct",
        ],
    ),
    TestPrompt(
        id="yielding",
        label="Yielding — shift moment",
        user="something just changed. I stopped fighting it and now it's different. I don't know what happened.",
        expected_qualities=[
            "acknowledges the shift",
            "does not over-explain",
            "supports the opening",
            "keeps response brief",
        ],
    ),
    TestPrompt(
        id="integration",
        label="Integration — after the experience",
        user="I think I understand what I saw but I don't know how to hold it. It feels too big.",
        expected_qualities=[
            "slower, more reflective tone",
            "does not rush to explain",
            "invites the person to find their own language",
            "acknowledges the significance",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def import_model_to_ollama() -> None:
    """Import the GGUF model into Ollama using the Modelfile."""
    if not Path(GGUF_PATH).exists():
        print(f"Error: GGUF file not found at {GGUF_PATH}")
        print("Run 2_convert.py first to produce the quantized model.")
        sys.exit(1)

    result = subprocess.run(
        ["ollama", "create", SHAMANOS_MODEL, "-f", MODELFILE_PATH],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error importing model into Ollama:\n{result.stderr}")
        sys.exit(1)

    print(f"Model imported as: {SHAMANOS_MODEL}")


def run_prompt(model: str, user_message: str) -> str:
    """Send a chat prompt to Ollama and return the response text."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 150,
        },
    }

    try:
        r = httpx.post(url, json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except httpx.ConnectError:
        print("Start Ollama with `ollama serve` before running this script")
        sys.exit(1)
    except httpx.TimeoutException:
        raise


def compare_responses(prompts: list[TestPrompt]) -> list[TestResult]:
    """Run all prompts against both models and print side-by-side comparisons."""
    results: list[TestResult] = []

    for prompt in prompts:
        print(f"\n{'=' * 70}")
        print(f"LABEL : {prompt.label}")
        print(f"USER  : {prompt.user}")
        print("-" * 70)

        # SHAMAN.OS (fine-tuned)
        try:
            shamanos_response = run_prompt(SHAMANOS_MODEL, prompt.user)
        except Exception as e:
            shamanos_response = f"ERROR: {e}"

        # Base model
        try:
            base_response = run_prompt(BASE_MODEL, prompt.user)
        except Exception as e:
            base_response = f"ERROR: {e}"

        print(f"SHAMAN.OS :\n{shamanos_response}")
        print(f"\nBASE ({BASE_MODEL}) :\n{base_response}")
        print("\nExpected qualities:")
        for quality in prompt.expected_qualities:
            print(f"  • {quality}")

        results.append(
            TestResult(
                prompt_id=prompt.id,
                label=prompt.label,
                user=prompt.user,
                shamanos_response=shamanos_response,
                base_response=base_response,
                expected_qualities=prompt.expected_qualities,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    return results


def save_results(results: list[TestResult], output_path: str = "test_results.json") -> None:
    """Write test results to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import_model_to_ollama()

    results = compare_responses(TEST_PROMPTS)

    save_results(results)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Prompts evaluated : {len(results)}")
    print(f"Models compared   : {SHAMANOS_MODEL} vs {BASE_MODEL}")
    print()
    print("EVALUATION GUIDE")
    print("-" * 70)
    print("For each prompt, check whether SHAMAN.OS meets the expected qualities.")
    print("A well-tuned model should:")
    print("  • Respond in short, present-tense sentences")
    print("  • Never dismiss or minimise what the person reports")
    print("  • Avoid forbidden phrases: 'hallucination', 'just', 'it's okay', 'don't worry'")
    print("  • Stay calm and grounded even at peak intensity")
    print()
    print(f"Full results written to test_results.json")
