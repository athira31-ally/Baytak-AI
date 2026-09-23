"""One command to build every artifact the API needs:

    python -m scripts.bootstrap            # listings, embeddings, simulated sessions, ranker, offline eval

Outputs (in ./artifacts): listings.pkl, embeddings.npz, local_embedder.joblib,
ranker.txt, metrics.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from app.config import get_settings
from app.data.reference import COMMUNITY_BY_NAME, resolve_hub
from app.data.synthetic import generate_listings, sample_user, true_utility
from app.recsys.embeddings import LocalEmbedder, get_embedder
from app.recsys.features import build_features
from app.recsys.ranker import LTRRanker
from app.recsys.retrieval import LocalRetriever
from app.schemas import UserQuery


def simulate_sessions(retriever: LocalRetriever, n_users: int, seed: int = 11, impressions: int = 50):
    rng = np.random.default_rng(seed)
    frames = []
    for uid in range(n_users):
        user = sample_user(rng)
        q = UserQuery(**user["query"])
        cand = retriever.retrieve(q, impressions)
        if len(cand) < 10:
            continue
        X = build_features(q, cand)
        hub = resolve_hub(q.work_location)
        u = np.array([true_utility(user, r, COMMUNITY_BY_NAME[r["community"].lower()], hub) for _, r in cand.iterrows()])
        z = (u - u.mean()) / (u.std() + 1e-9) + rng.normal(0, 0.5, len(u))
        grade = np.select([z > 1.8, z > 1.2, z > 0.6], [3, 2, 1], 0)   # contact / save / click / none
        X = X.assign(qid=uid, label=grade, listing_id=cand["listing_id"].to_numpy())
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


def evaluate(df: pd.DataFrame, scores: np.ndarray, k: int = 10) -> dict:
    df = df.assign(_s=scores)
    ndcg, prec, hit = [], [], []
    for _, g in df.groupby("qid"):
        if g["label"].max() == 0:
            continue
        ndcg.append(ndcg_score([g["label"].to_numpy()], [g["_s"].to_numpy()], k=k))
        top = g.nlargest(k, "_s")
        prec.append((top["label"] >= 2).mean())
        hit.append(float((top["label"] == 3).any()))
    return {f"ndcg@{k}": round(float(np.mean(ndcg)), 4), f"precision@{k}(save+)": round(float(np.mean(prec)), 4),
            f"hit_rate@{k}(contact)": round(float(np.mean(hit)), 4), "queries": len(ndcg)}


def main(n_listings: int, n_users: int) -> None:
    s = get_settings()
    t0 = time.time()
    print(f"[1/5] Generating {n_listings} listings")
    listings = generate_listings(n_listings)
    listings.to_pickle(s.data_dir / "listings.pkl")

    print("[2/5] Embedding listing descriptions")
    embedder = get_embedder(s)
    if isinstance(embedder, LocalEmbedder):
        embedder.fit(listings["description"].tolist())
    vectors = embedder.embed(listings["description"].tolist())
    np.savez(s.data_dir / "embeddings.npz", vectors=vectors, embedder=embedder.name)

    print(f"[3/5] Simulating {n_users} user sessions")
    data = simulate_sessions(LocalRetriever(listings, vectors, embedder), n_users)
    qids = data["qid"].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(qids)
    n = len(qids)
    split = {"train": set(qids[: int(.8 * n)]), "val": set(qids[int(.8 * n): int(.9 * n)]), "test": set(qids[int(.9 * n):])}
    parts = {k: data[data["qid"].isin(v)].sort_values("qid") for k, v in split.items()}
    groups = {k: v.groupby("qid", sort=True).size().tolist() for k, v in parts.items()}

    print("[4/5] Training LightGBM LambdaRank")
    ranker = LTRRanker(s.data_dir / "ranker.txt")
    importance = ranker.fit(parts["train"], parts["train"]["label"].to_numpy(), groups["train"],
                            parts["val"], parts["val"]["label"].to_numpy(), groups["val"])

    print("[5/5] Offline evaluation on held-out users")
    test = parts["test"]
    metrics = {
        "embedder": embedder.name,
        "n_listings": n_listings, "n_sessions": int(n), "rows": len(data),
        "ranker": evaluate(test, ranker.score(test)),
        "baseline_retrieval": evaluate(test, test["retrieval_score"].to_numpy()),
        "random": evaluate(test, np.random.default_rng(1).random(len(test))),
        "feature_importance_gain": {k: round(float(v), 1) for k, v in list(importance.items())[:10]},
        "seconds": round(time.time() - t0, 1),
    }
    (s.data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("ranker", "baseline_retrieval", "random")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", type=int, default=3000)
    ap.add_argument("--users", type=int, default=2500)
    a = ap.parse_args()
    main(a.listings, a.users)
