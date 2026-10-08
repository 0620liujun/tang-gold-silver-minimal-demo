# Relation-classification minimal demo

This repository is a synthetic-data example for the relation classifier. It shows the file format, the four-node graph rule, and how to call the model code.

Running it does not reproduce the scores reported in the paper. The four bundled sentences were written for this demo. They are not the research corpus and they are not a train, validation, or test split.

The pre-annotation prompt is not included.

A local demo of the Fig. 8 GraphRAG question-answering flow is in `graphrag_demo/`. Fig. 8 is the research prototype interface. This repository does not provide a public web server. The demo uses a small sample graph rather than the copyrighted research data. See `graphrag_demo/README.md` for the DeepSeek API key limitation.

```bash
python -m pip install -r graphrag_demo/requirements.txt
python graphrag_demo/run_graphrag_demo.py
```

Retrieval runs locally without an API key. `python graphrag_demo/run_graphrag_demo.py --use-llm` calls DeepSeek only when `DEEPSEEK_API_KEY` is set by the user. The demo does not reproduce the retrieval or question-answering metrics reported in the paper.

## Run

```bash
python scripts/run_demo.py
```

This uses only the Python standard library. It checks `sample_data/relation_samples.jsonl` and prints the graph edges for each pair.

One record looks like this:

```text
{"id":"demo-001","text":"演示用银盒呈圆形，盖面饰有莲瓣纹。","entity1":"演示用银盒","entity2":"莲瓣纹","head_type":"器物","tail_type":"纹样","relation":"外观装饰","synthetic":true}
```

Entity and relation names are listed in `schema/labels.json`.

## Model entry point

The network is ERNIE 3.0 Base Chinese (`ernie-3.0-base-zh`) plus BiLSTM or LSTM and GAT or GCN. No checkpoint is stored in this repository. Evaluation requires a checkpoint you supply:

```bash
python scripts/train_relation.py --eval_only --checkpoint PATH_TO_YOUR_CHECKPOINT --test_file sample_data/relation_samples.jsonl
```

`scripts/train_relation.py` is an architecture entry point. It does not implement the paper training procedure. It has no fixed random seed, no train/validation/test protocol, no mini-batch loader, no best-checkpoint selection, no full paper metric suite, and no hardware record.

```bash
python scripts/train_relation.py --train --output checkpoints/demo.pdparams --epochs 1
```

That command writes a new file only to the path you give `--output`. Do not commit it. A run on these four sentences cannot stand in for the paper result.

Install the model stack only when you need that entry point. `requirements.txt` names the target versions: `paddlepaddle==3.3.0` (CPU), `paddlenlp==2.8.1`, and `numpy==1.26.4`. The default is the CPU build, from the official Paddle 3.3 CPU index:

```bash
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddlenlp==2.8.1 numpy==1.26.4
```

A GPU build is optional. Install it only when the NVIDIA driver and CUDA version match, and do not install it beside the CPU package. Pick one index from the Paddle 3.3 documentation:

```bash
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

These commands specify the target environment. They are not a record that training has been run in this repository.

## Not included

- Research sentences and official data splits
- Annotation files, source books, reports, and museum text
- Knowledge-graph exports, database files, vector indexes, and embedding caches
- Checkpoints, optimizer state, logs, API keys, and local absolute paths
- `__pycache__/` and `*.pyc`
- The pre-annotation prompt

### Data access restrictions

The full research dataset contains text derived from copyrighted scholarly books, archaeological reports, museum publications, and other protected sources. Because these materials are subject to the copyright and licensing conditions of the original rights holders, the source-derived corpus and related research data cannot be publicly redistributed through this repository.

Accordingly, the complete corpus, formal dataset splits, and full knowledge graph are not included. To support methodological and peer-review verification, this repository instead provides synthetic examples, configuration files, runnable code, and minimal workflow demonstrations covering the main processing, relation-classification, graph-construction, and GraphRAG procedures.
