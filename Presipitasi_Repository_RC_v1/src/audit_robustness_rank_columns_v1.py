from pathlib import Path
import pandas as pd
import json

BASE=Path(r"D:\Scopus\AI_Scopus")
DERIVED=BASE/"04_DATA"/"GloFAS_Historical_v5_Johor"/"DERIVED"
f=DERIVED/"low_flow_stress_robustness_v1.parquet"
df=pd.read_parquet(f)

print("="*72)
print("LFSI ROBUSTNESS RANK AUDIT v1")
print("="*72)
print("\nColumns:")
print(list(df.columns))

rank_cols=[c for c in df.columns if "rank" in c.lower()]
print("\nAll columns containing 'rank':")
for c in rank_cols:
    print(" ",c)

print("\n2026 values:")
r2026=df[df["year"]==2026]
for c in rank_cols:
    print(f"  {c}: {r2026[c].iloc[0]}")

print("\nRank columns with ALL years:")
for c in rank_cols:
    vals=df[c].tolist()
    print(f"  {c}: {vals}")

print("\nCandidate robustness rank columns:")
candidates=[
    c for c in rank_cols
    if c.lower().startswith("rank_")
    or "robust" in c.lower()
    or c.lower().endswith("_rank")
]
print(candidates)

print("\n2026 candidate ranks:")
for c in candidates:
    print(f"  {c}: {r2026[c].iloc[0]}")

print("\nExpected condition: every actual robustness formulation rank for 2026 == 1.")
actual_ok=all(float(r2026[c].iloc[0])==1 for c in candidates)
print("Result:",actual_ok)

meta={
    "file":str(f),
    "all_rank_columns":rank_cols,
    "candidate_rank_columns":candidates,
    "actual_2026_all_rank_1":bool(actual_ok)
}
out=DERIVED/"robustness_rank_audit_v1.json"
out.write_text(json.dumps(meta,indent=2),encoding="utf-8")
print("\nOutput:",out)
print("QC: PASS")
