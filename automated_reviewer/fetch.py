"""Fetch a sample of ICLR papers + ground-truth decisions from OpenReview.

This is the dataset step. It mirrors the paper's evaluation setup: ICLR is the
only top venue that publishes every accept/reject decision, so it gives us
ground truth to score the automated reviewer against.

Run:
    python -m automated_reviewer.fetch --n 60 --venue ICLR.cc/2024/Conference

Output: data/papers.json -- a list of records with full paper text and the
binary ground-truth decision.

OpenReview public data is readable anonymously. Optional credentials:
    export OPENREVIEW_USERNAME=...   export OPENREVIEW_PASSWORD=...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cval(content: dict, key: str, default=""):
    """Read a field from an OpenReview note's content.

    API v2 wraps every field as {"value": ...}; v1 stores it bare.
    """
    v = content.get(key)
    if isinstance(v, dict):
        return v.get("value", default)
    return v if v is not None else default


def _reply_invitations(reply: dict) -> list:
    """Return the invitation id(s) of a reply, across API v1 and v2."""
    invs = reply.get("invitations")
    if isinstance(invs, list):
        return invs
    inv = reply.get("invitation")
    return [inv] if inv else []


def _extract_decision(submission) -> str | None:
    """Find the raw decision string in a submission's direct replies."""
    replies = (submission.details or {}).get("directReplies", []) or []
    for reply in replies:
        if not any(inv.endswith("/Decision") for inv in _reply_invitations(reply)):
            continue
        content = reply.get("content", {}) or {}
        for key in ("decision", "recommendation"):
            if key in content:
                val = content[key]
                return val.get("value") if isinstance(val, dict) else val
    return None


def _binarize(raw_decision: str) -> str:
    """Map a raw decision string to 'Accept' or 'Reject'."""
    return "Accept" if raw_decision.strip().lower().startswith("accept") else "Reject"


def _pdf_to_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def get_client():
    import openreview

    return openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.environ.get("OPENREVIEW_USERNAME"),
        password=os.environ.get("OPENREVIEW_PASSWORD"),
    )


def fetch(venue: str, n: int, balanced: bool, out_path: str, seed: int,
          pdf_dir: str | None) -> None:
    client = get_client()

    _log(f"Fetching submissions for {venue} ...")
    submissions = client.get_all_notes(
        invitation=f"{venue}/-/Submission", details="directReplies"
    )
    _log(f"  {len(submissions)} submissions returned")

    # Keep only submissions with a published decision.
    candidates = []
    for sub in submissions:
        raw = _extract_decision(sub)
        if raw is None:
            continue
        candidates.append((sub, raw, _binarize(raw)))
    _log(f"  {len(candidates)} have a published decision")
    if not candidates:
        _log("No decisions found. Decisions may not be public for this venue yet.")
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if balanced:
        # interleave accepts and rejects so we draw an even mix as we walk
        accepts = [c for c in candidates if c[2] == "Accept"]
        rejects = [c for c in candidates if c[2] == "Reject"]
        order = []
        for a, r in zip(accepts, rejects):
            order.extend([a, r])
        candidates = order

    papers = []
    if pdf_dir:
        os.makedirs(pdf_dir, exist_ok=True)

    for sub, raw, decision in candidates:
        if len(papers) >= n:
            break
        content = sub.content or {}
        title = _cval(content, "title")
        abstract = _cval(content, "abstract")
        try:
            pdf_bytes = client.get_attachment(id=sub.id, field_name="pdf")
            full_text = _pdf_to_text(pdf_bytes)
        except Exception as e:  # noqa: BLE001 - skip unreadable papers, keep going
            _log(f"  skip {sub.id}: PDF unavailable ({e})")
            continue
        if len(full_text) < 1000:
            _log(f"  skip {sub.id}: extracted text too short ({len(full_text)} chars)")
            continue

        if pdf_dir:
            with open(os.path.join(pdf_dir, f"{sub.id}.pdf"), "wb") as fh:
                fh.write(pdf_bytes)

        papers.append({
            "id": sub.id,
            "venue": venue,
            "title": title,
            "abstract": abstract,
            "raw_decision": raw,
            "decision": decision,
            "n_chars": len(full_text),
            "full_text": full_text,
        })
        _log(f"  [{len(papers)}/{n}] {decision:7s} {title[:60]}")

    n_accept = sum(1 for p in papers if p["decision"] == "Accept")
    _log(f"\nCollected {len(papers)} papers "
         f"({n_accept} Accept / {len(papers) - n_accept} Reject)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(papers, fh, indent=2)
    _log(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch ICLR papers from OpenReview.")
    ap.add_argument("--venue", default="ICLR.cc/2024/Conference",
                    help="OpenReview venue id (default: ICLR.cc/2024/Conference)")
    ap.add_argument("--n", type=int, default=60,
                    help="Number of papers to collect (default: 60)")
    ap.add_argument("--balanced", action="store_true",
                    help="Draw a roughly 50/50 accept/reject mix")
    ap.add_argument("--out", default="data/papers.json",
                    help="Output JSON path (default: data/papers.json)")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    ap.add_argument("--save-pdfs", metavar="DIR", default=None,
                    help="Also save raw PDFs to this directory")
    args = ap.parse_args()
    fetch(args.venue, args.n, args.balanced, args.out, args.seed, args.save_pdfs)


if __name__ == "__main__":
    main()
