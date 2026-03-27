"""
0_audit.py — SHAMAN.OS Fine-Tune Pipeline: Dataset Auditor

Validates a JSONL dataset for structural integrity, content quality,
and training suitability before any compute is spent.

Usage:
    python 0_audit.py <dataset.jsonl> [audit_report.json]
"""

import json
import sys
import os
import re
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AuditFlag:
    index: int
    check: str
    severity: str   # "error" | "warning" | "info"
    message: str
    detail: dict


@dataclass
class FieldLengthStats:
    min_words: int
    max_words: int
    mean_words: float
    median_words: float
    too_short: list  # indices with < 10 words
    too_long: list   # indices with > 300 words


@dataclass
class AuditReport:
    total_records: int
    flags: list
    field_length_stats: FieldLengthStats
    duplicate_pairs: list  # list[tuple[int, int]]
    summary: dict          # dict[str, int] — count per check name


# ---------------------------------------------------------------------------
# Per-record check functions
# ---------------------------------------------------------------------------

EXPECTED_ROLES = ["system", "user", "assistant"]


def check_structure(record: dict, idx: int) -> list:
    """
    Returns error AuditFlags for any structural violation in the record.
    Returns empty list for a perfectly-formed record.

    Checks:
    - 'messages' key exists
    - messages is a list
    - exactly 3 items
    - each item has 'role' and 'content' keys
    - roles are exactly [system, user, assistant] in that order
    """
    flags = []

    if "messages" not in record:
        flags.append(AuditFlag(
            index=idx,
            check="structure",
            severity="error",
            message="Missing 'messages' key",
            detail={"keys_found": list(record.keys())}
        ))
        return flags

    messages = record["messages"]

    if not isinstance(messages, list):
        flags.append(AuditFlag(
            index=idx,
            check="structure",
            severity="error",
            message="'messages' is not a list",
            detail={"type": type(messages).__name__}
        ))
        return flags

    if len(messages) != 3:
        flags.append(AuditFlag(
            index=idx,
            check="structure",
            severity="error",
            message=f"Expected exactly 3 messages, found {len(messages)}",
            detail={"count": len(messages)}
        ))
        return flags

    for i, item in enumerate(messages):
        if not isinstance(item, dict):
            flags.append(AuditFlag(
                index=idx,
                check="structure",
                severity="error",
                message=f"Message at position {i} is not a dict",
                detail={"position": i, "type": type(item).__name__}
            ))
            continue
        if "role" not in item:
            flags.append(AuditFlag(
                index=idx,
                check="structure",
                severity="error",
                message=f"Message at position {i} missing 'role' key",
                detail={"position": i, "keys_found": list(item.keys())}
            ))
        if "content" not in item:
            flags.append(AuditFlag(
                index=idx,
                check="structure",
                severity="error",
                message=f"Message at position {i} missing 'content' key",
                detail={"position": i, "keys_found": list(item.keys())}
            ))

    # If any key-level errors were found, skip role-order check
    if flags:
        return flags

    actual_roles = [m["role"] for m in messages]
    if actual_roles != EXPECTED_ROLES:
        flags.append(AuditFlag(
            index=idx,
            check="structure",
            severity="error",
            message=f"Roles must be [system, user, assistant], found {actual_roles}",
            detail={"expected": EXPECTED_ROLES, "found": actual_roles}
        ))

    return flags


def check_empty_fields(record: dict, idx: int) -> list:
    """
    Returns warning AuditFlags for any content field that is empty or whitespace-only.
    Only runs if structure is valid (messages key exists with 3 items).
    """
    flags = []

    try:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            return flags

        for i, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            content = item.get("content", None)
            if content is None:
                continue
            if not isinstance(content, str) or content.strip() == "":
                role = item.get("role", f"position_{i}")
                flags.append(AuditFlag(
                    index=idx,
                    check="empty_fields",
                    severity="warning",
                    message=f"Empty or whitespace-only content in '{role}' message",
                    detail={"position": i, "role": role}
                ))
    except Exception:
        pass

    return flags


# Literary analysis contamination phrases to detect in the USER field
_LITERARY_PHRASES = [
    "the passage",
    "the text",
    "the author",
    "the scripture",
    "this section",
    "the narrator",
    "the writing",
    "describes",
    "according to",
    "the document",
    "in the book",
    "the passage describes",
    "as described",
]


def check_literary_analysis(record: dict, idx: int) -> list:
    """
    Checks the USER field for literary-analysis contamination phrases.
    Returns an info flag if ANY of the phrases are found (case-insensitive).
    """
    flags = []

    try:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return flags

        # User is at index 1 in [system, user, assistant]
        user_msg = messages[1]
        if not isinstance(user_msg, dict):
            return flags

        content = user_msg.get("content", "")
        if not isinstance(content, str):
            return flags

        content_lower = content.lower()
        found = [phrase for phrase in _LITERARY_PHRASES if phrase in content_lower]

        if found:
            flags.append(AuditFlag(
                index=idx,
                check="literary_analysis",
                severity="info",
                message="User field contains literary-analysis language",
                detail={"phrases_found": found}
            ))
    except Exception:
        pass

    return flags


# First-person indicators to look for in the USER field
_FIRST_PERSON_INDICATORS = [
    "i ",
    "i'm",
    "i've",
    "i can",
    "i don't",
    "i feel",
    "i see",
    "i hear",
    "i think",
    "i am",
    "my ",
    "something is",
    "there is",
    "it's",
    "it keeps",
]


def check_first_person(record: dict, idx: int) -> list:
    """
    Checks the USER field for first-person voice indicators.
    Returns an info flag if NONE of the indicators appear (user lacks first-person voice).
    """
    flags = []

    try:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return flags

        # User is at index 1 in [system, user, assistant]
        user_msg = messages[1]
        if not isinstance(user_msg, dict):
            return flags

        content = user_msg.get("content", "")
        if not isinstance(content, str):
            return flags

        content_lower = content.lower()
        has_first_person = any(indicator in content_lower for indicator in _FIRST_PERSON_INDICATORS)

        if not has_first_person:
            flags.append(AuditFlag(
                index=idx,
                check="first_person",
                severity="info",
                message="User field lacks first-person voice",
                detail={"indicators_checked": _FIRST_PERSON_INDICATORS}
            ))
    except Exception:
        pass

    return flags


def check_sequence_length(record: dict, idx: int, max_tokens: int = 512) -> list:
    """
    Estimates token count using word-count approximation:
        estimated_tokens = len(full_text.split()) * 1.3
    where full_text is all content fields concatenated.

    Returns a warning flag if estimated tokens > max_tokens.
    """
    flags = []

    try:
        messages = record.get("messages")
        if not isinstance(messages, list):
            return flags

        parts = []
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)

        full_text = " ".join(parts)
        word_count = len(full_text.split())
        estimated_tokens = word_count * 1.3

        if estimated_tokens > max_tokens:
            flags.append(AuditFlag(
                index=idx,
                check="sequence_length",
                severity="warning",
                message=f"Estimated token count ({estimated_tokens:.0f}) exceeds {max_tokens}",
                detail={
                    "word_count": word_count,
                    "estimated_tokens": round(estimated_tokens, 1),
                    "max_tokens": max_tokens,
                }
            ))
    except Exception:
        pass

    return flags


# ---------------------------------------------------------------------------
# Aggregate checks
# ---------------------------------------------------------------------------

def check_field_lengths(records: list) -> FieldLengthStats:
    """
    Computes word-count statistics across all assistant content fields.
    Returns a FieldLengthStats with min, max, mean, median, too_short, too_long.
    """
    word_counts = []
    too_short = []
    too_long = []

    for idx, record in enumerate(records):
        try:
            messages = record.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 3:
                continue
            assistant_msg = messages[2]
            if not isinstance(assistant_msg, dict):
                continue
            content = assistant_msg.get("content", "")
            if not isinstance(content, str):
                continue
            wc = len(content.split())
            word_counts.append(wc)
            if wc < 10:
                too_short.append(idx)
            if wc > 300:
                too_long.append(idx)
        except Exception:
            continue

    if not word_counts:
        return FieldLengthStats(
            min_words=0,
            max_words=0,
            mean_words=0.0,
            median_words=0.0,
            too_short=[],
            too_long=[],
        )

    return FieldLengthStats(
        min_words=min(word_counts),
        max_words=max(word_counts),
        mean_words=statistics.mean(word_counts),
        median_words=statistics.median(word_counts),
        too_short=too_short,
        too_long=too_long,
    )


def check_duplicates(records: list, threshold: float = 0.70) -> list:
    """
    O(n²) Jaccard overlap on user-field tokens.
    Each pair (i, j) with i < j is reported at most once.
    Returns warning AuditFlags for pairs exceeding the threshold.
    """
    flags = []

    def get_user_tokens(record):
        try:
            messages = record.get("messages", [])
            if isinstance(messages, list) and len(messages) >= 2:
                user_msg = messages[1]
                if isinstance(user_msg, dict):
                    content = user_msg.get("content", "")
                    if isinstance(content, str):
                        return set(re.findall(r"\w+", content.lower()))
        except Exception:
            pass
        return set()

    user_token_sets = [get_user_tokens(r) for r in records]

    for i in range(len(user_token_sets)):
        for j in range(i + 1, len(user_token_sets)):
            tokens_i = user_token_sets[i]
            tokens_j = user_token_sets[j]
            denom = max(len(tokens_i), len(tokens_j))
            if denom == 0:
                continue
            overlap = len(tokens_i & tokens_j) / denom
            if overlap > threshold:
                flags.append(AuditFlag(
                    index=j,
                    check="duplicate",
                    severity="warning",
                    message=f">{threshold * 100:.0f}% overlap with record {i}",
                    detail={"pair": [i, j], "overlap": round(overlap, 4)}
                ))

    return flags


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_audit(dataset_path: str, output_path: str = "audit_report.json") -> AuditReport:
    """
    Loads JSONL line-by-line, runs all 7 validation checks, assembles AuditReport,
    writes audit_report.json, and prints a human-readable summary.
    """
    records = []
    flags = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                flags.append(AuditFlag(
                    index=line_idx,
                    check="json_parse",
                    severity="error",
                    message=f"JSON parse error: {e}",
                    detail={"line": line[:120]}
                ))
                records.append({})  # placeholder to keep indices aligned

    for idx, record in enumerate(records):
        if not record:
            continue
        flags.extend(check_structure(record, idx))
        flags.extend(check_empty_fields(record, idx))
        flags.extend(check_literary_analysis(record, idx))
        flags.extend(check_first_person(record, idx))
        flags.extend(check_sequence_length(record, idx))

    field_stats = check_field_lengths(records)
    dup_flags = check_duplicates(records)
    flags.extend(dup_flags)

    duplicate_pairs = [tuple(f.detail["pair"]) for f in dup_flags if "pair" in f.detail]

    summary: dict = {}
    for flag in flags:
        summary[flag.check] = summary.get(flag.check, 0) + 1

    report = AuditReport(
        total_records=len(records),
        flags=flags,
        field_length_stats=field_stats,
        duplicate_pairs=duplicate_pairs,
        summary=summary,
    )

    # Serialize to JSON
    def _serialize(obj):
        if isinstance(obj, AuditReport):
            return {
                "total_records": obj.total_records,
                "flags": [asdict(f) for f in obj.flags],
                "field_length_stats": asdict(obj.field_length_stats),
                "duplicate_pairs": [list(p) for p in obj.duplicate_pairs],
                "summary": obj.summary,
            }
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(_serialize(report), out, indent=2)

    _print_summary(report)
    return report


def _print_summary(report: AuditReport) -> None:
    total_flags = len(report.flags)
    errors = sum(1 for f in report.flags if f.severity == "error")
    warnings = sum(1 for f in report.flags if f.severity == "warning")
    infos = sum(1 for f in report.flags if f.severity == "info")

    print(f"\n{'='*60}")
    print(f"  SHAMAN.OS Audit Report")
    print(f"{'='*60}")
    print(f"  Total records : {report.total_records}")
    print(f"  Total flags   : {total_flags}  (errors={errors}, warnings={warnings}, info={infos})")
    print(f"\n  Per-check summary:")
    for check_name, count in sorted(report.summary.items()):
        print(f"    {check_name:<25} {count}")
    stats = report.field_length_stats
    print(f"\n  Field-length stats (assistant words):")
    print(f"    min={stats.min_words}  max={stats.max_words}  "
          f"mean={stats.mean_words:.1f}  median={stats.median_words:.1f}")
    print(f"    too_short (<10 words): {len(stats.too_short)} records")
    print(f"    too_long  (>300 words): {len(stats.too_long)} records")
    print(f"  Duplicate pairs: {len(report.duplicate_pairs)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 0_audit.py <dataset.jsonl> [audit_report.json]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "audit_report.json"

    if not Path(dataset_path).exists():
        print(f"Error: dataset file not found: {dataset_path}")
        sys.exit(1)

    run_audit(dataset_path, output_path)
    print(f"Audit report written to: {output_path}")
