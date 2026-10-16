"""Shared setup: model IDs, prompts, chat formatting (Qwen3 thinking mode OFF)."""
import os

SPECDEC = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(SPECDEC, "hf_cache"))   # keep models in this folder
if os.path.isdir(os.path.join(os.environ["HF_HOME"], "hub")):         # offline once models are downloaded
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

TARGET = "mlx-community/Qwen3-8B-4bit"
DRAFT = "mlx-community/Qwen3-0.6B-4bit"

TYPO_PARAGRAPH = (
    "Speculative decodng is a tecnique for making large langauge models generate text faster "
    "without changing what they say. A small, cheap draft model guesses the next few tokens, and "
    "the large target model checks all of those guesses in a single forward pass. Because a forward "
    "pass at batch size one is limited by memroy bandwidth rather than arithmetic, checking five "
    "tokens costs almost the same as checking one. When the guesses are right, the model skips ahead "
    "several tokens at once; when they are wrong, it simply keeps the corect prefix and continues. "
    "The output is mathematicaly identical to what the big model would have produced on it's own."
)

CODE_TO_EDIT = '''def moving_average(lst, window):
    result = []
    for i in range(len(lst) - window + 1):
        chunk = lst[i:i + window]
        result.append(sum(chunk) / window)
    return result


def normalize(lst):
    lo, hi = min(lst), max(lst)
    if hi == lo:
        return [0.0 for _ in lst]
    return [(x - lo) / (hi - lo) for x in lst]
'''

PROMPTS = {
    "code": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. "
            "Include a docstring and three example test cases using assert statements.",
    "rewrite": "Fix the spelling mistakes in the following paragraph. Return only the corrected "
               "paragraph, nothing else.\n\n" + TYPO_PARAGRAPH,
    "creative": "Write a short, original story (about 200 words) about a lighthouse keeper who "
                "discovers that the lighthouse lamp is alive.",
    "math": "Solve step by step: A train leaves city A at 9:00 AM at 60 km/h. Another train leaves "
            "city B, 300 km away, at 10:00 AM toward city A at 90 km/h. At what time do they meet, "
            "and how far from city A? Show each step.",
    "code_edit": "Add type hints and a one-line docstring to each function below, and rename the "
                 "parameter `lst` to `numbers`. Return the full updated code only.\n\n```python\n"
                 + CODE_TO_EDIT + "```",
}


def chat_tokens(tokenizer, text):
    """Qwen3 chat template with thinking DISABLED (template emits an empty <think></think>)."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, "keys"):          # transformers 5 may return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def load_pair():
    from mlx_lm import load
    target, tok = load(TARGET)
    draft, dtok = load(DRAFT)
    assert dtok.vocab_size == tok.vocab_size, "draft and target must share a tokenizer"
    return target, draft, tok
