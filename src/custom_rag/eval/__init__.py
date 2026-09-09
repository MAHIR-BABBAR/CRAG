"""Evaluation package: IR metrics, datasets, and golden-set runner."""

from custom_rag.eval.datasets import PreparedCorpus, prepare_scifact
from custom_rag.eval.metrics import (
    MetricResult,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    summarize_metrics,
)
from custom_rag.eval.report import report_to_markdown, write_report
from custom_rag.eval.runner import (
    EvalQuery,
    ModeEvalReport,
    evaluate_mode,
    load_qrels,
    run_benchmark_eval,
    run_eval_suite,
)

__all__ = [
    "EvalQuery",
    "MetricResult",
    "ModeEvalReport",
    "PreparedCorpus",
    "evaluate_mode",
    "hit_at_k",
    "load_qrels",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "prepare_scifact",
    "recall_at_k",
    "report_to_markdown",
    "run_benchmark_eval",
    "run_eval_suite",
    "summarize_metrics",
    "write_report",
]
