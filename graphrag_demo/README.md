# Fig. 8 GraphRAG 本地 demo

Fig. 8 展示的是本文实现的研究原型界面：唐代金银器知识图谱可视化与智能问答。当前没有提供公开在线 Web server。

本目录提供 Fig. 8 所对应核心 GraphRAG 问答流程的本地运行 demo：

问题 → 实体 / 关系检索 → 图谱事实检索 → 证据检索 → 答案生成

完整研究数据由于第三方出版物版权限制不能公开。这里只提供少量样例数据，用于展示知识图谱结构、检索流程和问答过程。样例句子是为 demo 编写的，不是文献原文，也不能代替论文中的实验结果。

检索在本地完成，不需要数据库、Neo4j 或 Web 服务。默认答案由检索到的三元组和证据用模板组织。原系统的答案生成使用 DeepSeek API；本仓库不提供 API key，也不在本地部署 DeepSeek。若要复现这一步，需要自行配置 DeepSeek API key。没有 key 时，图谱检索、关系检索和证据返回仍然会执行。

## 运行

在本目录执行：

```bash
pip install -r requirements.txt
python run_graphrag_demo.py
```

`requirements.txt` 不安装第三方包。本地检索和模板答案只使用 Python 标准库。

程序会回答三个固定问题，并依次输出：

- Question
- Retrieved entities
- Retrieved triples
- Evidence
- Answer

也可以只问一个问题：

```bash
python run_graphrag_demo.py --question "鎏金舞马衔杯纹皮囊式银壶有哪些结构部件？"
```

## DeepSeek

需要自行申请 key，并设置为环境变量 `DEEPSEEK_API_KEY`。不要把 key 写入文件或提交到仓库。

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python run_graphrag_demo.py --use-llm
```

bash：

```bash
export DEEPSEEK_API_KEY=your_key
python run_graphrag_demo.py --use-llm
```

调用地址默认为 `https://api.deepseek.com`，模型默认为 `deepseek-chat`。未设置 key 或请求失败时，程序仍输出检索结果，并用模板生成答案。

## 样例数据

`sample_kg.json` 每条记录包含：

- `source_entity`
- `relation`
- `target_entity`
- `evidence`
- `source`

样例围绕 Fig. 8 中的问题“介绍一下鎏金舞马衔杯纹皮囊式银壶”，包含壶身、壶腹、覆莲瓣式壶盖、银环、银链、衔杯舞马形象、模压、捶击等节点，以及一条对照器物，用来区分检索结果。其中“皮囊壶造型”只表示形制来源。

## English

Fig. 8 shows the research prototype interface implemented in this paper. No public web server is provided.

This directory is a local demo of that GraphRAG flow: question, entity and relation retrieval, graph-fact retrieval, evidence retrieval, and answer generation. The full research data cannot be released because of third-party publication copyright. `sample_kg.json` is a small illustrative sample, not the research corpus and not a reproduction of the reported scores.

```bash
pip install -r requirements.txt
python run_graphrag_demo.py
```

To use DeepSeek for the answer step, set `DEEPSEEK_API_KEY` yourself and run:

```bash
python run_graphrag_demo.py --use-llm
```

Without a key, graph retrieval, relation retrieval, and evidence lookup still run, and a template assembles the answer.
