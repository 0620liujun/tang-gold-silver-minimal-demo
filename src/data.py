# -*- coding: utf-8 -*-
"""Load the public relation-classification JSONL format.

The demo accepts only a small file. It does not name or recognize any
research split, corpus export, or intermediate annotation file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

MAX_DEMO_RECORDS = 32


def load_relation_jsonl(path: Path, max_records: int = MAX_DEMO_RECORDS) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if len(rows) >= max_records:
                raise SystemExit(
                    f"This demo reads at most {max_records} records. "
                    "Use a small file prepared for the public example."
                )
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            rows.append(item)
    if not rows:
        raise SystemExit("No records found in the input file.")
    return rows


def check_relation_row(row: Dict, relation_types: List[str], entity_types: List[str]) -> List[str]:
    errors = []
    text = row.get("text") or ""
    if not text:
        errors.append("missing text")
    for key in ("entity1", "entity2", "relation"):
        if not row.get(key):
            errors.append(f"missing {key}")
    if row.get("relation") and row["relation"] not in relation_types:
        errors.append(f"illegal relation: {row.get('relation')}")
    for key, type_key in (("entity1", "head_type"), ("entity2", "tail_type")):
        mention = row.get(key) or ""
        if mention and text and mention not in text:
            errors.append(f"{key} not found in text")
        etype = row.get(type_key)
        if etype and etype not in entity_types:
            errors.append(f"illegal {type_key}: {etype}")
    return errors
