# -*- coding: utf-8 -*-
"""Train or evaluate the relation classifier on a user-provided JSONL file.

The task checkpoint is not part of this repository. Evaluation requires
--checkpoint. Training downloads the public ERNIE 3.0 Base Chinese encoder
and writes a new checkpoint only to the path you pass with --output.
Running this script on sample_data does not reproduce the paper metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_relation_jsonl
from src.graph_rules import find_span


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal ERNIE relation-classification entry point")
    parser.add_argument("--train_file", default="", help="User-provided JSONL. Defaults to the synthetic sample when training.")
    parser.add_argument("--test_file", default="", help="User-provided JSONL used for --eval_only.")
    parser.add_argument("--checkpoint", default="", help="Path to a checkpoint you provide. Not bundled.")
    parser.add_argument("--output", default="", help="Where to write a newly trained checkpoint.")
    parser.add_argument("--ernie_model_name", default="ernie-3.0-base-zh")
    parser.add_argument("--sequence_type", default="bilstm", choices=["bilstm", "lstm"])
    parser.add_argument("--graph_type", default="gat", choices=["gat", "gcn"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def relation_to_id(labels):
    return {name: index for index, name in enumerate(labels)}


def encode_batch(rows, tokenizer, label_ids, max_length):
    import paddle

    texts, labels, positions = [], [], []
    for row in rows:
        texts.append(row["text"])
        labels.append(label_ids[row["relation"]])
        span1 = find_span(row["text"], row["entity1"])
        span2 = find_span(row["text"], row["entity2"])
        positions.append((span1, span2))
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pd")
    input_ids = encoded["input_ids"]
    if "attention_mask" in encoded:
        attention_mask = encoded["attention_mask"]
    else:
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        attention_mask = (input_ids != pad_id).astype("int64")
    token_positions = []
    for index, (span1, span2) in enumerate(positions):
        seq_len = int(input_ids[index].shape[0])
        length = max(1, len(texts[index]))

        def to_token(span):
            start = min(seq_len - 2, max(1, int(span[0] / length * seq_len)))
            end = min(seq_len - 2, max(start, int(span[1] / length * seq_len)))
            return start, end

        token_positions.append((to_token(span1), to_token(span2)))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "entity_positions": token_positions,
        "labels": paddle.to_tensor(labels, dtype="int64"),
    }


def main() -> int:
    args = parse_args()
    labels = json.loads((ROOT / "schema" / "labels.json").read_text(encoding="utf-8"))["relation_types"]
    label_ids = relation_to_id(labels)

    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit(
                "Evaluation needs --checkpoint. This repository does not include a trained checkpoint; "
                "supply one you are allowed to use."
            )
        test_path = Path(args.test_file) if args.test_file else ROOT / "sample_data" / "relation_samples.jsonl"
        rows = load_relation_jsonl(test_path)
    elif args.train:
        train_path = Path(args.train_file) if args.train_file else ROOT / "sample_data" / "relation_samples.jsonl"
        rows = load_relation_jsonl(train_path)
        if not args.output:
            raise SystemExit("Training needs --output for the new checkpoint path. Do not commit that file.")
        print("[warn] synthetic or user-supplied files here are not the paper splits and will not reproduce paper scores.")
    else:
        raise SystemExit("Choose --train or --eval_only. For the offline format demo, run scripts/run_demo.py.")

    from src.model import ERNIESeqGraphModel, compute_metrics, load_checkpoint, load_ernie

    encoder, tokenizer = load_ernie(args.ernie_model_name)
    model = ERNIESeqGraphModel(
        ernie_model=encoder,
        num_relations=len(labels),
        sequence_type=args.sequence_type,
        graph_type=args.graph_type,
    )
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    import paddle
    import paddle.nn as nn

    if args.train:
        optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters(), weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(args.epochs):
            batch = encode_batch(rows, tokenizer, label_ids, args.max_length)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["entity_positions"])
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            print(f"epoch {epoch + 1} loss={float(loss.item()):.4f}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        paddle.save(model.state_dict(), args.output)
        print(f"wrote {args.output}")
        return 0

    model.eval()
    batch = encode_batch(rows, tokenizer, label_ids, args.max_length)
    with paddle.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"], batch["entity_positions"])
    prediction = logits.argmax(axis=-1).numpy()
    metrics = compute_metrics(batch["labels"].numpy(), prediction, len(labels))
    print(json.dumps({"n": len(rows), **metrics}, ensure_ascii=False))
    print("[info] scores on the bundled sample are format checks, not paper metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
