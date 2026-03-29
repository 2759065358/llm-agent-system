"""
chunk 策略对比实验：100 / 300 / 500

示例：
python scripts/chunk_strategy_experiment.py \
  --content_file README.md \
  --query "这个项目的核心能力是什么" \
  --query "系统架构是什么" \
  --top_k 3
"""

from __future__ import annotations

import argparse
import json
import time
import re
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag.rag_pipeline import SimpleRAGPipeline


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(t) > 1]


def keyword_hit_score(query: str, retrieved_docs: List[str]) -> float:
    q = set(tokenize(query))
    d = set(tokenize("\n".join(retrieved_docs)))
    if not q:
        return 0.0
    return len(q & d) / len(q)


def run_once(content: str, query: str, strategy: str, top_k: int) -> Dict:
    collection_name = f"rag_chunk_{strategy}"
    pipe = SimpleRAGPipeline(collection_name=collection_name, chunk_strategy=strategy)
    pipe.add_document(content)

    start = time.perf_counter()
    docs = pipe.retrieve(query, top_k=top_k, enable_rerank=True)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "strategy": strategy,
        "query": query,
        "latency_ms": round(latency_ms, 2),
        "score": round(keyword_hit_score(query, docs), 4),
        "top_docs": docs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_file", type=str, default="")
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", type=str, default="logs/chunk_experiment_results.json")
    args = parser.parse_args()

    if args.content_file:
        with open(args.content_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = args.content

    if not content.strip():
        raise ValueError("请提供 --content_file 或 --content")

    strategies = ["100", "300", "500"]

    all_results: List[Dict] = []
    for strategy in strategies:
        for q in args.query:
            all_results.append(run_once(content, q, strategy, args.top_k))

    summary = {}
    for strategy in strategies:
        cur = [r for r in all_results if r["strategy"] == strategy]
        summary[strategy] = {
            "avg_latency_ms": round(sum(x["latency_ms"] for x in cur) / len(cur), 2),
            "avg_score": round(sum(x["score"] for x in cur) / len(cur), 4),
        }

    report = {
        "summary": summary,
        "details": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✅ 已输出实验结果到: {args.output}")


if __name__ == "__main__":
    main()
