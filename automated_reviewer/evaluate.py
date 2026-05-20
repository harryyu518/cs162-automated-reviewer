"""Score the automated reviewer against ground-truth ICLR decisions.

Joins results/reviews.json (predictions) with data/papers.json (ground truth)
and reports the Table-1 metrics from the paper: balanced accuracy, accuracy,
F1, AUC, FPR, FNR -- alongside two trivial baselines (always-reject, random).

Run:
    python -m automated_reviewer.evaluate
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from . import metrics


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _row(label: str, m: dict) -> str:
    return (f"{label:<22} "
            f"{m['balanced_accuracy']:>9.3f} "
            f"{m['accuracy']:>9.3f} "
            f"{m['f1']:>9.3f} "
            f"{m['auc']:>9.3f} "
            f"{m['fpr']:>9.3f} "
            f"{m['fnr']:>9.3f}")


def evaluate(papers_path: str, reviews_path: str, out_path: str,
             seed: int) -> None:
    with open(papers_path) as fh:
        papers = {p["id"]: p for p in json.load(fh)}
    with open(reviews_path) as fh:
        reviews = json.load(fh)

    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    skipped = 0

    for pid, rec in reviews.items():
        final = rec.get("final")
        if pid not in papers or final is None:
            skipped += 1
            continue
        y_true.append(1 if papers[pid]["decision"] == "Accept" else 0)
        y_pred.append(1 if final["decision"] == "Accept" else 0)
        # AUC needs a continuous score; fall back to the decision if missing
        ov = final.get("overall")
        scores.append(float(ov) if ov is not None else float(y_pred[-1]))

    if not y_true:
        _log("No scored papers found. Run the reviewer first.")
        sys.exit(1)

    reviewer = metrics.compute_all(y_true, y_pred, scores)

    # Baselines, for context (cf. Table 1).
    always_reject = metrics.compute_all(y_true, [0] * len(y_true), scores)
    rng = random.Random(seed)
    rand_pred = [rng.randint(0, 1) for _ in y_true]
    rand_scores = [rng.random() for _ in y_true]
    random_dec = metrics.compute_all(y_true, rand_pred, rand_scores)

    # Score distribution for the report.
    overalls = [final.get("overall") for pid, rec in reviews.items()
                if (final := rec.get("final")) and final.get("overall") is not None]
    mean_overall = sum(overalls) / len(overalls) if overalls else 0.0

    n = reviewer["n"]
    ci = reviewer["balanced_accuracy_95ci"]
    print()
    print(f"Evaluated {n} papers "
          f"({reviewer['n_accept']} Accept / {reviewer['n_reject']} Reject), "
          f"{skipped} skipped")
    print(f"Mean predicted overall score: {mean_overall:.2f}")
    print()
    print(f"{'Reviewer':<22} {'bal.acc':>9} {'acc':>9} {'F1':>9} "
          f"{'AUC':>9} {'FPR':>9} {'FNR':>9}")
    print("-" * 84)
    print(_row("Automated reviewer", reviewer))
    print(_row("Baseline: always rej", always_reject))
    print(_row("Baseline: random", random_dec))
    print("-" * 84)
    print(f"Balanced-accuracy 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    cm = reviewer["confusion"]
    print(f"Confusion: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    print()

    report = {
        "n_papers": n,
        "n_skipped": skipped,
        "mean_predicted_overall": mean_overall,
        "automated_reviewer": reviewer,
        "baseline_always_reject": always_reject,
        "baseline_random": random_dec,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    _log(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the automated reviewer.")
    ap.add_argument("--papers", default="data/papers.json",
                    help="Ground-truth papers JSON (default: data/papers.json)")
    ap.add_argument("--reviews", default="results/reviews.json",
                    help="Reviewer output JSON (default: results/reviews.json)")
    ap.add_argument("--out", default="results/metrics.json",
                    help="Metrics output JSON (default: results/metrics.json)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the random baseline")
    args = ap.parse_args()
    evaluate(args.papers, args.reviews, args.out, args.seed)


if __name__ == "__main__":
    main()
