#!/bin/bash
# Stock mlx_lm CLI, two runs: big model alone, then big model + draft model.
# Watch the "Generation: ... tokens-per-sec" line at the end of each run.
#   bash side_by_side.sh            # code prompt
#   bash side_by_side.sh rewrite    # copy-heavy prompt (biggest speedup)
HERE="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="${HF_HOME:-$HERE/../hf_cache}"   # models live here (falls back to download if missing)

if [ "$1" == "rewrite" ]; then
  PROMPT="Fix the spelling mistakes in the following paragraph. Return only the corrected paragraph, nothing else.

Speculative decodng is a tecnique for making large langauge models generate text faster without changing what they say. A small, cheap draft model guesses the next few tokens, and the large target model checks all of those guesses in a single forward pass. Because a forward pass at batch size one is limited by memroy bandwidth rather than arithmetic, checking five tokens costs almost the same as checking one. When the guesses are right, the model skips ahead several tokens at once; when they are wrong, it simply keeps the corect prefix and continues. The output is mathematicaly identical to what the big model would have produced on it's own."
  N=4
else
  PROMPT="Write a Python function merge_intervals(intervals) that merges overlapping intervals. Include a docstring and three example test cases using assert statements."
  N=2
fi

echo "================  BIG MODEL ALONE  (Qwen3-8B, 4-bit)  ================"
python3 -m mlx_lm generate --model mlx-community/Qwen3-8B-4bit \
  --prompt "$PROMPT" --max-tokens 256 --temp 0 \
  --chat-template-config '{"enable_thinking": false}'

echo
echo "================  + DRAFT MODEL  (Qwen3-0.6B guesses $N tokens ahead)  ================"
python3 -m mlx_lm generate --model mlx-community/Qwen3-8B-4bit \
  --draft-model mlx-community/Qwen3-0.6B-4bit --num-draft-tokens $N \
  --prompt "$PROMPT" --max-tokens 256 --temp 0 \
  --chat-template-config '{"enable_thinking": false}'
