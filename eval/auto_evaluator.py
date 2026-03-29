from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import re


STOPWORDS = {
    "的", "了", "和", "是", "在", "就", "都", "而", "及", "与",
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]+", text) if t.lower() not in STOPWORDS and len(t) > 1]


def _f1(pred: str, truth: str) -> float:
    p = _tokenize(pred)
    t = _tokenize(truth)

    if not p or not t:
        return 0.0

    p_set, t_set = set(p), set(t)
    overlap = len(p_set & t_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(p_set)
    recall = overlap / len(t_set)
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvalResult:
    question: str
    answer: str
    accuracy: float
    mode: str
    reason: str
    ts: str


class AutoEvaluator:
    """
    自动评估模块。

    输入:
    - question
    - answer
    - reference_answer(可选)

    输出:
    - accuracy(0~1)
    - 结构化日志
    """

    def __init__(self, log_path: str = "logs/eval_logs.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        question: str,
        answer: str,
        reference_answer: Optional[str] = None,
    ) -> Dict:
        if reference_answer and reference_answer.strip():
            score = _f1(answer, reference_answer)
            mode = "reference_f1"
            reason = "使用 reference_answer 做 token-level F1"
        else:
            # 无参考答案时用 question coverage 作为弱监督分数
            q_tokens = set(_tokenize(question))
            a_tokens = set(_tokenize(answer))
            if not q_tokens:
                score = 0.0
            else:
                score = len(q_tokens & a_tokens) / len(q_tokens)
            mode = "question_coverage"
            reason = "未提供 reference_answer，使用 question 关键词覆盖率近似准确率"

        result = EvalResult(
            question=question,
            answer=answer,
            accuracy=round(float(score), 4),
            mode=mode,
            reason=reason,
            ts=datetime.utcnow().isoformat(),
        )

        self._append_log(result)
        return asdict(result)

    def _append_log(self, result: EvalResult):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
