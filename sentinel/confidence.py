"""Explainable confidence scoring — the autonomy gate.

SENTINEL never deploys on gut feeling. Each resolution is scored from
independent, auditable signals; the full breakdown ships with the verdict so
a human (or an auditor, later) can see exactly *why* the agent was allowed —
or not allowed — to act on its own.

Score = weighted sum of component scores in [0, 1]. Resolutions at or above
`settings.confidence_threshold` deploy autonomously; anything below pauses
for human approval.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.config import settings

WEIGHTS = {
    "tests": 0.40,        # did the full suite pass in the sandbox?
    "agent": 0.25,        # the agent's own calibrated confidence
    "iterations": 0.20,   # fewer fix attempts -> more confident diagnosis
    "blast_radius": 0.15, # smaller patches are safer to auto-deploy
}


@dataclass
class Verdict:
    score: float
    threshold: float
    autonomous: bool
    components: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "threshold": self.threshold,
            "autonomous": self.autonomous,
            "components": self.components,
        }


def assess(
    tests_passed: bool,
    agent_confidence: float,
    iterations_used: int,
    files_touched: int,
    lines_changed: int,
) -> Verdict:
    """Combine independent signals into one explainable score."""
    agent_confidence = min(max(agent_confidence, 0.0), 1.0)

    tests_score = 1.0 if tests_passed else 0.0

    # 1 attempt -> 1.0, each extra attempt costs 25%, floor at 0.
    iter_score = max(0.0, 1.0 - 0.25 * max(iterations_used - 1, 0))

    # 1 file & <=10 lines is ideal; penalize breadth and size.
    size_penalty = 0.15 * max(files_touched - 1, 0) + 0.02 * max(lines_changed - 10, 0)
    blast_score = max(0.0, 1.0 - size_penalty)

    components = {
        "tests": {
            "score": tests_score,
            "weight": WEIGHTS["tests"],
            "reason": "full test suite passed in sandbox" if tests_passed else "tests not passing",
        },
        "agent": {
            "score": agent_confidence,
            "weight": WEIGHTS["agent"],
            "reason": f"agent self-reported confidence {agent_confidence:.2f}",
        },
        "iterations": {
            "score": round(iter_score, 3),
            "weight": WEIGHTS["iterations"],
            "reason": f"fix converged in {iterations_used} attempt(s)",
        },
        "blast_radius": {
            "score": round(blast_score, 3),
            "weight": WEIGHTS["blast_radius"],
            "reason": f"{files_touched} file(s), {lines_changed} line(s) changed",
        },
    }

    score = sum(c["score"] * c["weight"] for c in components.values())
    return Verdict(
        score=score,
        threshold=settings.confidence_threshold,
        autonomous=score >= settings.confidence_threshold,
        components=components,
    )
