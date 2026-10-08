# -*- coding: utf-8 -*-
"""Four-node candidate graph used by the relation classifier.

Nodes are fixed:
0 entity1, 1 entity2, 2 between-context, 3 global-context.

This module uses character offsets so the demo can run without Paddle.
The training script maps the same spans onto tokenizer positions before
calling the neural model.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

NODE_NAMES = ("entity1", "entity2", "between_context", "global_context")


def find_span(text: str, entity: str) -> Tuple[int, int]:
    start = text.find(entity) if entity else -1
    if start < 0:
        mid = len(text) // 2
        return max(0, mid - 5), max(0, mid - 1)
    return start, start + len(entity) - 1


def build_adjacency(text: str, entity1: str, entity2: str) -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int]]:
    span1 = find_span(text, entity1)
    span2 = find_span(text, entity2)
    adj = [[0, 0, 0, 0] for _ in range(4)]
    for i in range(4):
        adj[i][i] = 1
    for src, dst in ((0, 3), (1, 3), (0, 2), (1, 2), (2, 3)):
        adj[src][dst] = 1
        adj[dst][src] = 1
    valid_len = max(2, len(text))
    center1 = (span1[0] + span1[1]) // 2
    center2 = (span2[0] + span2[1]) // 2
    if abs(center1 - center2) <= max(2, valid_len // 6):
        adj[0][1] = 1
        adj[1][0] = 1
    return adj, span1, span2


def edge_list(adj: Sequence[Sequence[int]]) -> List[str]:
    edges = []
    for i, row in enumerate(adj):
        for j, flag in enumerate(row):
            if flag and i < j:
                edges.append(f"{NODE_NAMES[i]} -- {NODE_NAMES[j]}")
    return edges
