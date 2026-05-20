"""Prompts for the automated reviewer.

Design notes
------------
The prompt is split into three parts so it caches well and stays faithful to
the paper (Lu et al., 2026, Methods -> The Automated Reviewer):

  REVIEWER_SYSTEM    -- short, frozen role statement
  REVIEW_GUIDELINES  -- long, frozen NeurIPS-style guidelines + scoring rubric
  build_user_prompt  -- the per-paper content (title, abstract, full text)

REVIEWER_SYSTEM + REVIEW_GUIDELINES form a stable prefix that is identical for
every paper, so it is sent once and cached (see providers.py). Only the
per-paper user prompt changes between requests.
"""

# --------------------------------------------------------------------------
# Stable prefix: role + guidelines (identical for every paper -> cacheable)
# --------------------------------------------------------------------------

REVIEWER_SYSTEM = (
    "You are an experienced AI researcher reviewing a paper submitted to a "
    "prestigious machine learning venue (NeurIPS / ICLR). You are rigorous, "
    "fair, and calibrated: you neither rubber-stamp weak work nor reject solid "
    "work for being unexciting. You base every judgement on evidence in the "
    "paper itself."
)

REVIEW_GUIDELINES = """\
# Reviewer guidelines

Follow the official NeurIPS reviewer guidelines. Produce a single structured
review. Read the whole paper before scoring. Be specific: cite sections,
equations, tables, or figures when you raise a point.

## What to assess

1. Soundness. Are the claims supported by correct theory and/or sufficient,
   well-designed experiments? Are baselines, datasets, metrics, and ablations
   appropriate? Are statistical comparisons fair (same budget, seeds, tuning)?
2. Presentation. Is the paper clearly written and well organized? Are figures
   and tables readable? Is prior work accurately described and cited?
3. Contribution. How novel and significant is the work relative to existing
   literature? Would the community benefit from it? Is the scope honest?

Also note: limitations the authors did or did not acknowledge, reproducibility
(is there enough detail or code to reproduce results?), and any ethical
concerns (data, human subjects, dual use, societal impact).

## Scoring rubric

Score every dimension. Use the FULL range — do not cluster around the middle.

soundness, presentation, contribution -- integer 1 to 4:
  4 = excellent   3 = good   2 = fair   1 = poor

overall -- integer 1 to 10:
  10 = award quality       9 = top 15% / oral
  8  = top 50% / accept    7 = good paper / accept
  6  = marginally above acceptance threshold
  5  = marginally below acceptance threshold
  4  = ok but not good enough / reject
  3  = clear reject        2 = strong reject        1 = trivial / wrong

confidence -- integer 1 to 5:
  5 = absolutely certain    4 = confident
  3 = fairly confident      2 = willing to defend but could be wrong
  1 = educated guess

decision -- "Accept" or "Reject":
  Choose "Accept" only if overall >= 6. Choose "Reject" if overall <= 5.
  The decision must be consistent with the overall score.

## Failure modes to watch for

Top-tier venues reject most submissions. Common reasons to reject: naive or
underdeveloped ideas, missing or weak baselines, claims unsupported by the
experiments, methodological errors, no ablations, overstated contributions,
and inaccurate or missing citations. Reward clear writing, honest scoping,
strong empirical or theoretical evidence, and genuine novelty.

## Output format

Respond with ONLY a single JSON object — no prose, no markdown, no code fences
before or after it. Use exactly these keys:

{
  "summary": "<2-4 sentence neutral summary of what the paper does>",
  "strengths": ["<strength>", "..."],
  "weaknesses": ["<weakness>", "..."],
  "questions": ["<question for the authors>", "..."],
  "limitations": "<are limitations / ethical concerns adequately addressed?>",
  "soundness": <int 1-4>,
  "presentation": <int 1-4>,
  "contribution": <int 1-4>,
  "overall": <int 1-10>,
  "confidence": <int 1-5>,
  "decision": "Accept" | "Reject"
}
"""

# --------------------------------------------------------------------------
# Area-chair meta-review (used when --n-reviews > 1)
# --------------------------------------------------------------------------

META_SYSTEM = (
    "You are an Area Chair at a prestigious machine learning venue. You are "
    "given several independent reviews of one paper. Weigh them — accounting "
    "for each reviewer's stated confidence — and decide the paper's fate."
)

META_GUIDELINES = """\
# Area-chair meta-review

You will be given the JSON reviews of a single paper from several reviewers.
Reconcile them into one final decision. Give more weight to higher-confidence
reviews and to concrete, evidence-based points over vague impressions.

Respond with ONLY a single JSON object — no prose, no code fences:

{
  "summary": "<1-2 sentence rationale for the final decision>",
  "soundness": <int 1-4>,
  "presentation": <int 1-4>,
  "contribution": <int 1-4>,
  "overall": <int 1-10>,
  "confidence": <int 1-5>,
  "decision": "Accept" | "Reject"
}

The decision must be consistent with overall (Accept iff overall >= 6).
"""


def build_user_prompt(title: str, abstract: str, full_text: str) -> str:
    """Per-paper content. This is the only part that varies between requests."""
    return (
        "Review the following paper.\n\n"
        f"## Title\n{title}\n\n"
        f"## Abstract\n{abstract}\n\n"
        f"## Full paper text\n{full_text}\n\n"
        "Now produce your review as a single JSON object, following the "
        "guidelines and output format exactly."
    )


def build_meta_prompt(reviews: list) -> str:
    """Per-paper content for the area-chair meta-review."""
    import json

    blocks = []
    for i, r in enumerate(reviews, 1):
        blocks.append(f"### Review {i}\n{json.dumps(r, indent=2)}")
    joined = "\n\n".join(blocks)
    return (
        f"Here are {len(reviews)} independent reviews of one paper.\n\n"
        f"{joined}\n\n"
        "Now produce your meta-review as a single JSON object."
    )
