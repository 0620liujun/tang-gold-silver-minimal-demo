# -*- coding: utf-8 -*-
"""Run the minimal demo on synthetic samples. No checkpoint and no API key."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import check_relation_row, load_relation_jsonl
from src.graph_rules import build_adjacency, edge_list


def main() -> int:
    labels = json.loads((ROOT / "schema" / "labels.json").read_text(encoding="utf-8"))
    rows = load_relation_jsonl(ROOT / "sample_data" / "relation_samples.jsonl")
    if not rows or len(rows) > 12:
        raise SystemExit("sample_data/relation_samples.jsonl must stay a small demo file")

    print(f"[demo] relation samples: {len(rows)}")
    for row in rows:
        errors = check_relation_row(row, labels["relation_types"], labels["entity_types"])
        if errors:
            raise SystemExit(f"{row.get('id')} failed: {'; '.join(errors)}")
        adj, span1, span2 = build_adjacency(row["text"], row["entity1"], row["entity2"])
        print(f"  {row['id']}  {row['entity1']} --{row['relation']}--> {row['entity2']}")
        print(f"    spans={span1},{span2}")
        print("    edges: " + "; ".join(edge_list(adj)))

    print("[demo] synthetic samples only. This run does not reproduce paper scores.")
    print("[demo] no checkpoint is bundled. Evaluation needs a user-provided --checkpoint.")
    print("[demo] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
