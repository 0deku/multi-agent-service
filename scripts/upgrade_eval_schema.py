import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_data_schema import normalize_example

TARGETS = [
    (Path("data/agent_eval.json"), "agent"),
    (Path("data/router_eval.json"), "router"),
    (Path("data/rag_eval.json"), "rag"),
    (Path("data/e2e_eval.json"), "e2e"),
    (Path("data/tool_eval.json"), "tool"),
]


def main():
    summary = {}
    for path, dataset in TARGETS:
        if not path.exists():
            summary[str(path)] = {"updated": 0, "skipped": "not_found"}
            continue

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        upgraded = [normalize_example(item, dataset=dataset) for item in raw]

        with path.open("w", encoding="utf-8") as f:
            json.dump(upgraded, f, ensure_ascii=False, indent=2)

        summary[str(path)] = {"updated": len(upgraded), "dataset": dataset}

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
