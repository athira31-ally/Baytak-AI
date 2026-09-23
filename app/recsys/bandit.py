"""Cold-start exploration with Thompson sampling.

New listings have no engagement history, so a pure ranker keeps burying them.
We reserve one slot per page for exploration: each fresh listing's
community is an arm with a Beta(successes+1, failures+1) posterior learned
from feedback; we sample and show the winner."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

POSITIVE = {"click", "save", "contact"}


class CommunityThompsonBandit:
    def __init__(self, seed: int | None = None):
        self.alpha: dict[str, float] = defaultdict(lambda: 1.0)
        self.beta: dict[str, float] = defaultdict(lambda: 1.0)
        self.rng = np.random.default_rng(seed)

    def update(self, community: str, event: str) -> None:
        if event in POSITIVE:
            self.alpha[community] += 1
        elif event in {"impression", "dismiss"}:
            self.beta[community] += 1 if event == "dismiss" else 0.2   # impressions are weak negatives

    def rebuild(self, events: list[dict], listing_to_community: dict[str, str]) -> None:
        self.alpha.clear(); self.beta.clear()
        for e in events:
            if e.get("source") == "explore" and e["listing_id"] in listing_to_community:
                self.update(listing_to_community[e["listing_id"]], e["event"])

    def pick(self, pool: pd.DataFrame, fresh_days: int = 14) -> pd.Series | None:
        """Choose one exploration candidate from listings not already on the page."""
        fresh = pool[pool["days_listed"] <= fresh_days]
        if fresh.empty:
            fresh = pool
        if fresh.empty:
            return None
        draws = {c: self.rng.beta(self.alpha[c], self.beta[c]) for c in fresh["community"].unique()}
        best = max(draws, key=draws.get)
        return fresh[fresh["community"] == best].iloc[0]

    def posterior(self) -> dict[str, dict]:
        keys = set(self.alpha) | set(self.beta)
        return {k: {"alpha": self.alpha[k], "beta": self.beta[k],
                    "mean": self.alpha[k] / (self.alpha[k] + self.beta[k])} for k in sorted(keys)}
