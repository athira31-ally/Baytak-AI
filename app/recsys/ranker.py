"""Stage 2 - learning-to-rank with LightGBM LambdaRank."""
from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.recsys.features import FEATURES

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")


class LTRRanker:
    def __init__(self, path: Path):
        self.path = path
        self.booster: lgb.Booster | None = lgb.Booster(model_file=str(path)) if path.exists() else None

    @property
    def ready(self) -> bool:
        return self.booster is not None

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: list[int],
            X_val: pd.DataFrame | None = None, y_val=None, groups_val=None) -> dict:
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                               num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.9, random_state=0, verbose=-1)
        kw = {}
        if X_val is not None:
            kw = dict(eval_set=[(X_val[FEATURES], y_val)], eval_group=[groups_val], eval_at=[5, 10],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        model.fit(X[FEATURES], y, group=groups, **kw)
        self.booster = model.booster_
        self.booster.save_model(str(self.path))
        imp = dict(zip(FEATURES, self.booster.feature_importance("gain")))
        return dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if not self.ready:
            return X["retrieval_score"].to_numpy()
        return self.booster.predict(X[FEATURES])
