"""Convert local PubMedQA data to JSONL for lm-eval custom task."""
import json
import os

with open("/root/dslm/pubmedqa/data/ori_pqal.json") as f:
    data = json.load(f)

out_dir = os.path.dirname(__file__)

items = []
for pid, v in data.items():
    items.append({
        "id": pid,
        "QUESTION": v["QUESTION"],
        "CONTEXTS": v["CONTEXTS"],
        "final_decision": v["final_decision"],
    })

n = len(items)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)

for split, start, end in [("train", 0, n_train), ("validation", n_train, n_train + n_valid), ("test", n_train + n_valid, n)]:
    path = os.path.join(out_dir, f"{split}.jsonl")
    with open(path, "w") as f:
        for item in items[start:end]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{split}: {end - start} -> {path}")

print(f"Total: {n}")
