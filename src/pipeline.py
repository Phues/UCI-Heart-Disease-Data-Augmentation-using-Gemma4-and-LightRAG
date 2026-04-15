import asyncio
import os
import sys
import pandas as pd
from rag_setup import build_rag, get_embedding_dim, check_ollama_reachable, wait_for_tunnel
from generator import get_constraints, generate_record, UCI_COLUMNS

OUTPUT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))
ORIGINAL_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/heart.csv"))

SEED_PROFILES = [
    {"age": 45, "sex": 1, "chol": 210, "trestbps": 125, "fbs": 0},  # mid-age male, moderate risk
    {"age": 55, "sex": 1, "chol": 245, "trestbps": 140, "fbs": 1},  # older male, high risk
    {"age": 60, "sex": 0, "chol": 280, "trestbps": 150, "fbs": 1},  # older female, high risk
    {"age": 40, "sex": 0, "chol": 190, "trestbps": 110, "fbs": 0},  # young female, low risk
    {"age": 70, "sex": 1, "chol": 300, "trestbps": 160, "fbs": 1},  # elderly male, very high risk
    {"age": 35, "sex": 1, "chol": 175, "trestbps": 118, "fbs": 0},  # young male, low risk
    {"age": 58, "sex": 0, "chol": 230, "trestbps": 135, "fbs": 0},  # post-menopausal female
    {"age": 50, "sex": 1, "chol": 260, "trestbps": 145, "fbs": 1},  # diabetic male
]

async def generate_dataset(n_records: int = 100) -> pd.DataFrame:
    records = []
    failed  = 0

    # Gate: tunnel must be up before we build RAG
    if not check_ollama_reachable():
        wait_for_tunnel("pipeline")

    dim = get_embedding_dim()
    rag = await build_rag(dim)

    print(f"\nGenerating {n_records} synthetic records via {os.getenv('OLLAMA_HOST', 'localhost')}...")

    try:
        for i in range(n_records):
            seed = SEED_PROFILES[i % len(SEED_PROFILES)].copy()
            seed["age"] = seed["age"] + (i % 7) - 3

            # Re-check tunnel health every 10 records
            if i % 10 == 0 and not check_ollama_reachable():
                wait_for_tunnel(f"record {i+1}")

            try:
                constraints = await get_constraints(rag, seed)
                record      = generate_record(seed, constraints)
                records.append(record)
                print(f"  [{i+1:>3}/{n_records}] age={record['age']:>2}  "
                      f"sex={record['sex']}  chol={record['chol']:>3}  "
                      f"trestbps={record['trestbps']:>3}  target={record['target']}")
            except Exception as e:
                failed += 1
                print(f"  [{i+1:>3}/{n_records}] FAILED: {e}")
    finally:
        await rag.finalize_storages()

    df = pd.DataFrame(records, columns=UCI_COLUMNS)
    print(f"\nDone: {len(records)} generated, {failed} failed.")
    return df

def compare_distributions(original: pd.DataFrame, synthetic: pd.DataFrame):
    print("\n=== Distribution Comparison ===")
    for col in ["age", "chol", "trestbps", "thalach"]:
        print(f"\n{col}:")
        print(f"  Original  — mean: {original[col].mean():.1f}  std: {original[col].std():.1f}  "
              f"min: {original[col].min()}  max: {original[col].max()}")
        print(f"  Synthetic — mean: {synthetic[col].mean():.1f}  std: {synthetic[col].std():.1f}  "
              f"min: {synthetic[col].min()}  max: {synthetic[col].max()}")
    print(f"\ntarget balance:")
    print(f"  Original:  {original['target'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  Synthetic: {synthetic['target'].value_counts(normalize=True).round(3).to_dict()}")

async def main(n_records: int = 50):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_synthetic = await generate_dataset(n_records)

    out_path = os.path.join(OUTPUT_DIR, "synthetic_heart_local.csv")
    df_synthetic.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    if os.path.exists(ORIGINAL_DATA):
        df_original = pd.read_csv(ORIGINAL_DATA, na_values="?")
        # UCI raw files use 'num' for the label; normalise to 'target'
        if "num" in df_original.columns and "target" not in df_original.columns:
            df_original = df_original.rename(columns={"num": "target"})
        # Binarise: original uses 0-4, we use 0/1
        if df_original["target"].max() > 1:
            df_original["target"] = (df_original["target"] > 0).astype(int)
        compare_distributions(df_original, df_synthetic)
    else:
        print(f"\nNo original data at {ORIGINAL_DATA} — skipping distribution comparison.")

    print("\nSynthetic dataset preview:")
    print(df_synthetic.head(10).to_string())

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    asyncio.run(main(n))