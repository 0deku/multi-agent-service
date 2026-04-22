import json
from pathlib import Path

TRAIN_SPLIT_PATH = Path("data/sft/planner_train_split.jsonl")
VALID_SPLIT_PATH = Path("data/sft/planner_valid_split.jsonl")

OUT_DIR = Path("data/sft/trainer")
TRAIN_OUT = OUT_DIR / "planner_train_trainer.jsonl"
VALID_OUT = OUT_DIR / "planner_valid_trainer.jsonl"


def main():
    train_rows = _read_jsonl(TRAIN_SPLIT_PATH)
    valid_rows = _read_jsonl(VALID_SPLIT_PATH)

    train_out_rows = [_to_trainer_row(r) for r in train_rows]
    valid_out_rows = [_to_trainer_row(r) for r in valid_rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(TRAIN_OUT, train_out_rows)
    _write_jsonl(VALID_OUT, valid_out_rows)

    summary = {
        "train_input": str(TRAIN_SPLIT_PATH),
        "valid_input": str(VALID_SPLIT_PATH),
        "train_output": str(TRAIN_OUT),
        "valid_output": str(VALID_OUT),
        "train_count": len(train_out_rows),
        "valid_count": len(valid_out_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _read_jsonl(path: Path):
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _to_trainer_row(row: dict):
    messages = row.get("messages") or []
    system = ""
    user = ""
    target = ""

    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        if role == "system":
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            target = content

    sample_weight = float((row.get("meta") or {}).get("sample_weight", 1.0))

    return {
        "input": {
            "system": system,
            "user": user,
        },
        "target": target,
        "weight": sample_weight,
    }


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
