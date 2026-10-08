# -*- coding: utf-8 -*-
"""ERNIE + sequence encoder + GAT/GCN relation classifier.

The public ERNIE 3.0 Base Chinese encoder is loaded by name from PaddleNLP.
Task checkpoints are not included. Pass a local file with --checkpoint.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GraphAttentionLayer(nn.Layer):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features, bias_attr=False)
        self.a = nn.Linear(2 * out_features, 1, bias_attr=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h: paddle.Tensor, adj: paddle.Tensor) -> paddle.Tensor:
        batch_size, num_nodes, _ = h.shape
        wh = self.W(h)
        pairs = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                pairs.append(paddle.concat([wh[:, i, :], wh[:, j, :]], axis=-1))
        stacked = paddle.stack(pairs, axis=1)
        scores = self.leakyrelu(self.a(stacked)).squeeze(-1).reshape([batch_size, num_nodes, num_nodes])
        masked = paddle.where(adj > 0, scores, paddle.full_like(scores, -1e9))
        attention = F.dropout(F.softmax(masked, axis=-1), p=self.dropout, training=self.training)
        return paddle.matmul(attention, wh)


class MultiHeadGraphAttentionLayer(nn.Layer):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.LayerList(
            [
                GraphAttentionLayer(in_features, out_features, dropout=dropout)
                for _ in range(max(1, int(num_heads)))
            ]
        )

    def forward(self, h: paddle.Tensor, adj: paddle.Tensor) -> paddle.Tensor:
        stacked = paddle.stack([head(h, adj) for head in self.heads], axis=0)
        return F.elu(stacked.mean(axis=0))


class GraphConvolutionLayer(nn.Layer):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, h: paddle.Tensor, adj: paddle.Tensor) -> paddle.Tensor:
        degree = adj.sum(axis=-1).clip(min=1.0)
        scale = paddle.pow(degree, -0.5)
        adj_norm = adj * scale.unsqueeze(-1) * scale.unsqueeze(-2)
        transformed = F.dropout(self.W(h), p=self.dropout, training=self.training)
        return F.elu(paddle.matmul(adj_norm, transformed))


class ERNIESeqGraphModel(nn.Layer):
    """Four-node relation classifier: entity1, entity2, between context, global context."""

    def __init__(
        self,
        ernie_model,
        num_relations: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        graph_layers: int = 2,
        graph_heads: int = 2,
        dropout: float = 0.1,
        sequence_type: str = "bilstm",
        graph_type: str = "gat",
    ):
        super().__init__()
        self.num_relations = num_relations
        self.sequence_type = sequence_type.lower()
        self.graph_type = graph_type.lower()
        if self.sequence_type not in ("bilstm", "lstm"):
            raise ValueError("sequence_type must be bilstm or lstm")
        if self.graph_type not in ("gat", "gcn"):
            raise ValueError("graph_type must be gat or gcn")

        self.ernie = ernie_model
        lstm_hidden = hidden_size if self.sequence_type == "bilstm" else hidden_size * 2
        direction = "bidirectional" if self.sequence_type == "bilstm" else "forward"
        self.lstm = nn.LSTM(
            input_size=self.ernie.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        graph_dim = hidden_size * 2
        self.seq_norm = nn.LayerNorm(graph_dim)
        self.seq_dropout = nn.Dropout(dropout)
        self.graph_layers = nn.LayerList()
        for _ in range(graph_layers):
            if self.graph_type == "gat":
                layer = MultiHeadGraphAttentionLayer(graph_dim, graph_dim, graph_heads, dropout)
            else:
                layer = GraphConvolutionLayer(graph_dim, graph_dim, dropout)
            self.graph_layers.append(layer)
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_norms = nn.LayerList([nn.LayerNorm(graph_dim) for _ in range(graph_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim * 2, graph_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relations),
        )
        self.graph_gain = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(2.0))

    def build_graph(self, entity_positions, seq_len: int, batch_size: int, attention_mask) -> paddle.Tensor:
        adj = paddle.zeros([batch_size, 4, 4], dtype="float32")
        for batch_idx in range(batch_size):
            span1, span2 = entity_positions[batch_idx]
            valid_len = int(attention_mask[batch_idx].sum().item()) if attention_mask is not None else seq_len
            valid_len = max(2, min(valid_len, seq_len))
            for i in range(4):
                adj[batch_idx, i, i] = 1.0
            for src, dst in ((0, 3), (1, 3), (0, 2), (1, 2), (2, 3)):
                adj[batch_idx, src, dst] = 1.0
                adj[batch_idx, dst, src] = 1.0
            center1 = (span1[0] + span1[1]) // 2
            center2 = (span2[0] + span2[1]) // 2
            if abs(center1 - center2) <= max(2, valid_len // 6):
                adj[batch_idx, 0, 1] = 1.0
                adj[batch_idx, 1, 0] = 1.0
        return adj

    def forward(self, input_ids, attention_mask, entity_positions):
        batch_size, seq_len = input_ids.shape
        sequence_output = self.seq_dropout(self.seq_norm(self.lstm(self.ernie(input_ids=input_ids, attention_mask=attention_mask)[0])[0]))
        entity1_features, entity2_features, between_features, global_features = [], [], [], []
        for batch_idx in range(batch_size):
            valid_len = max(2, min(int(attention_mask[batch_idx].sum().item()), seq_len))
            span1, span2 = entity_positions[batch_idx]

            def span_mean(start, end, fallback):
                start = max(1, min(int(start), valid_len - 2))
                end = max(start, min(int(end), valid_len - 2))
                return sequence_output[batch_idx, start : end + 1].mean(axis=0), start, end

            entity1, e1s, e1e = span_mean(span1[0], span1[1], None)
            entity2, e2s, e2e = span_mean(span2[0], span2[1], None)
            mid_start = min(e1e, e2e) + 1
            mid_end = max(e1s, e2s) - 1
            between = sequence_output[batch_idx, mid_start : mid_end + 1].mean(axis=0) if mid_start <= mid_end else (entity1 + entity2) / 2
            entity1_features.append(entity1)
            entity2_features.append(entity2)
            between_features.append(between)
            global_features.append(sequence_output[batch_idx, 1 : valid_len - 1].mean(axis=0))

        entity1_features = paddle.stack(entity1_features)
        entity2_features = paddle.stack(entity2_features)
        node_features = paddle.stack(
            [entity1_features, entity2_features, paddle.stack(between_features), paddle.stack(global_features)],
            axis=1,
        )
        adj = self.build_graph(entity_positions, seq_len, batch_size, attention_mask)
        for index, layer in enumerate(self.graph_layers):
            residual = node_features
            node_features = self.graph_norms[index](residual + self.graph_dropout(layer(node_features, adj)))
        gain = F.softplus(self.graph_gain)
        entity1_final = entity1_features + gain * (node_features[:, 0, :] - entity1_features)
        entity2_final = entity2_features + gain * (node_features[:, 1, :] - entity2_features)
        return self.classifier(paddle.concat([entity1_final, entity2_final], axis=-1))


def load_ernie(model_name: str = "ernie-3.0-base-zh"):
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer

    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    encoder = ErnieModel.from_pretrained(model_name)
    return encoder, tokenizer


def load_checkpoint(model: ERNIESeqGraphModel, checkpoint: str) -> None:
    state = paddle.load(checkpoint)
    model.set_state_dict(state)
    print(f"loaded checkpoint: {checkpoint}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    f1_scores = []
    for label in range(num_classes):
        tp = np.sum((y_true == label) & (y_pred == label))
        predicted = np.sum(y_pred == label)
        actual = np.sum(y_true == label)
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / actual) if actual else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "f1_macro": float(np.mean(f1_scores)) if f1_scores else 0.0}
