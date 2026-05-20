"""Automated paper reviewer — a recreation of the Automated Reviewer from
Lu et al., "Towards end-to-end automation of AI research" (Nature, 2026).

Pipeline:
  fetch    -> pull ICLR papers + ground-truth decisions from OpenReview
  review   -> prompt an LLM to act as a conference reviewer (JSON scores)
  evaluate -> compare predicted accept/reject against ground truth
"""

# Load API keys from a .env file in the working directory, if present, so the
# providers can read them via os.environ. No-op if python-dotenv isn't
# installed or there is no .env file.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except ImportError:
    pass

