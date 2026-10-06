#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="/tmp/latent-overlay:$repo/benchmarks/latent"
export HF_HUB_OFFLINE=1
cd /tmp
for blocks in 32 11000; do
  /tmp/latent/.venv/bin/python "$repo/benchmarks/latent/check_engine.py" \
    --invariant --blocks "$blocks" \
    --output "$repo/results/optimization/preemption-$blocks.json" \
    > "$repo/results/optimization/preemption-$blocks.log" 2>&1
done
/tmp/latent/.venv/bin/python - "$repo/results/optimization" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
small = json.loads((out / "preemption-32.json").read_text())
large = json.loads((out / "preemption-11000.json").read_text())
assert len(small["tokens"]) == len(large["tokens"]) == 12
result = {
    "exact_requests": sum(a == b for a, b in zip(small["tokens"], large["tokens"])),
    "requests": 12,
    "small_cache_preemptions": sum(small["preemptions"]),
    "large_cache_preemptions": sum(large["preemptions"]),
}
(out / "preemption-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
assert result["exact_requests"] == 12
assert result["small_cache_preemptions"] > 0
assert result["large_cache_preemptions"] == 0
PY
