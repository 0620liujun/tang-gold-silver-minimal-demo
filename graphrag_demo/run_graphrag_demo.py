# -*- coding: utf-8 -*-
"""Local demo of the Fig. 8 GraphRAG question-answering flow.

question -> entity / relation retrieval -> graph facts -> evidence -> answer

Default (no API key):

    python run_graphrag_demo.py

Optional DeepSeek answer generation. The key is read only from the environment
and is never printed. If it is missing or the call fails, the template answer
is used and retrieval still runs.

    python run_graphrag_demo.py --use-llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_KG = ROOT / "sample_kg.json"
FIELDS = ("source_entity", "relation", "target_entity", "evidence", "source")

DEMO_QUESTIONS = (
    "介绍一下鎏金舞马衔杯纹皮囊式银壶",
    "鎏金舞马衔杯纹皮囊式银壶有哪些结构部件？",
    "鎏金舞马衔杯纹皮囊式银壶的外观装饰和制作工艺是什么？",
)

# Longer cues are listed first. A cue selects a relation type; it is not an answer.
RELATION_CUES = (
    ("结构部件与连接", ("结构部件与连接", "结构部件", "部件", "组成")),
    ("外观装饰", ("外观装饰", "装饰", "纹样")),
    ("制作工艺", ("制作工艺", "工艺")),
    ("功能用途", ("功能用途", "用途", "功能")),
    ("材料构成", ("材料构成", "材料", "材质")),
    ("基本信息", ("基本信息", "造型", "形制")),
    ("文化象征", ("文化象征", "象征")),
    ("连接方式", ("连接方式",)),
)

HOPS = 1


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_kg(path: Path) -> list:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("sample_kg.json must contain a non-empty list.")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise SystemExit(f"sample_kg.json row {index} is not an object.")
        missing = [field for field in FIELDS if not str(row.get(field) or "").strip()]
        if missing:
            raise SystemExit(f"sample_kg.json row {index} is missing {', '.join(missing)}.")
    return rows


def entity_names(rows: list) -> list:
    names = {row["source_entity"] for row in rows}
    names.update(row["target_entity"] for row in rows)
    return sorted(names, key=len, reverse=True)


def retrieve_entities(question: str, rows: list) -> list:
    """Match entity strings in the question.

    A shorter name inside a longer matched name is not a separate entity.
    """
    occupied = []
    found = []
    for name in entity_names(rows):
        start = 0
        while True:
            index = question.find(name, start)
            if index < 0:
                break
            end = index + len(name)
            overlaps = any(not (end <= left or index >= right) for left, right in occupied)
            if overlaps:
                start = index + 1
                continue
            occupied.append((index, end))
            if name not in found:
                found.append(name)
            start = end
    return found


def retrieve_relations(question: str) -> list:
    matched = []
    for relation, cues in RELATION_CUES:
        if any(cue in question for cue in cues):
            matched.append(relation)
    return matched


def retrieve_triples(rows: list, entities: list, relations: list, hops: int = HOPS) -> list:
    """Keep triples in the neighborhood of the matched entities."""
    if not entities:
        return []
    nodes = set(entities)
    selected = []
    seen = set()
    for _ in range(max(0, hops)):
        added = False
        for index, row in enumerate(rows):
            if index in seen:
                continue
            if row["source_entity"] not in nodes and row["target_entity"] not in nodes:
                continue
            if relations and row["relation"] not in relations:
                continue
            selected.append(row)
            seen.add(index)
            nodes.add(row["source_entity"])
            nodes.add(row["target_entity"])
            added = True
        if not added:
            break
    return selected


def retrieve_evidence(triples: list) -> list:
    evidence = []
    seen = set()
    for row in triples:
        key = (row["evidence"], row["source"])
        if key in seen:
            continue
        seen.add(key)
        evidence.append({"evidence": row["evidence"], "source": row["source"]})
    return evidence


def template_answer(triples: list) -> str:
    if not triples:
        return "当前样例图谱中没有检索到足以回答该问题的证据。"
    parts = []
    for row in triples:
        parts.append(
            f"{row['relation']}：{row['source_entity']}—{row['target_entity']}。"
            f"证据：{row['evidence']}（{row['source']}）"
        )
    return "根据样例知识图谱检索结果：\n" + "\n".join(parts)


def build_llm_context(triples: list) -> str:
    lines = []
    for row in triples:
        lines.append(
            f"{row['source_entity']} | {row['relation']} | {row['target_entity']}\n"
            f"evidence: {row['evidence']}\nsource: {row['source']}"
        )
    return "\n\n".join(lines)


def call_deepseek(question: str, triples: list) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "只根据用户给出的图谱三元组和证据回答。"
                    "证据不足时明确说明证据不足，不要补充证据之外的事实。"
                    "回答中保留关系、证据和来源。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：\n{question}\n\n检索结果：\n{build_llm_context(triples)}",
            },
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    text = text.strip()
    if not text:
        raise RuntimeError("DeepSeek returned an empty answer")
    return text


def answer_question(question: str, triples: list, use_llm: bool) -> tuple:
    fallback = template_answer(triples)
    if not use_llm:
        return fallback, "template"
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return fallback, "template; DeepSeek API key is not set"
    try:
        return call_deepseek(question, triples), "DeepSeek"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, KeyError):
        return fallback, "template; DeepSeek request failed"


def run_question(question: str, rows: list, use_llm: bool) -> dict:
    entities = retrieve_entities(question, rows)
    relations = retrieve_relations(question)
    triples = retrieve_triples(rows, entities, relations)
    evidence = retrieve_evidence(triples)
    answer, mode = answer_question(question, triples, use_llm)
    return {
        "question": question,
        "entities": entities,
        "relations": relations,
        "triples": triples,
        "evidence": evidence,
        "answer": answer,
        "mode": mode,
    }


def print_result(result: dict) -> None:
    print("Question")
    print(result["question"])
    print("Retrieved entities")
    if result["entities"]:
        for name in result["entities"]:
            print(f"- {name}")
    else:
        print("(none)")
    print("Retrieved triples")
    if result["triples"]:
        for row in result["triples"]:
            print(f"- {row['source_entity']} | {row['relation']} | {row['target_entity']}")
    else:
        print("(none)")
    print("Evidence")
    if result["evidence"]:
        for item in result["evidence"]:
            print(f"- {item['evidence']} ({item['source']})")
    else:
        print("(none)")
    print("Answer")
    print(f"({result['mode']})")
    print(result["answer"])
    print("")


def parse_args():
    parser = argparse.ArgumentParser(description="Local Fig. 8 GraphRAG demo")
    parser.add_argument("--question", default="", help="Ask one question instead of the built-in examples.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate the answer with DeepSeek when DEEPSEEK_API_KEY is set.",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    rows = load_kg(SAMPLE_KG)
    if args.question.strip():
        questions = (args.question.strip(),)
    else:
        questions = DEMO_QUESTIONS
    for question in questions:
        print_result(run_question(question, rows, args.use_llm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
