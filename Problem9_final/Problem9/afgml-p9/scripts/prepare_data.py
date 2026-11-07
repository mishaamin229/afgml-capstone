import os, sys, yaml, pandas as pd
sys.path.insert(0, ".")
from src.data import prepare, group_denominators

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/data.yaml"))
df = pd.read_csv(cfg["source_file"], sep=cfg["sep"])
splits = prepare(df, cfg)
os.makedirs('data/processed', exist_ok=True); os.makedirs('results/metrics', exist_ok=True)
rows = []
for name, fr in splits.items():
    fr.to_csv(f"data/processed/{name}.csv", index=False)
    d = group_denominators(fr, cfg["min_group_support"]); d.insert(0, "split", name); rows.append(d)
pd.concat(rows).to_csv("results/metrics/group_denominators.csv", index=False)
print({k: (len(v), round(v.y.mean(), 4)) for k, v in splits.items()})
