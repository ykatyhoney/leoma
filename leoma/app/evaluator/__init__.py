"""Self-evaluation: a validator evaluates only the tasks it sampled (no cross-validation)."""

from leoma.app.evaluator.main import evaluate_sampled_task


__all__ = ["evaluate_sampled_task"]
