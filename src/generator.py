import asyncio
import json
import re
import os
import ollama
from lightrag import QueryParam
from rag_setup import build_rag

OLLAMA_MODEL = "gemma4:e4b"

# UCI Heart Disease dataset columns
UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs",
               "restecg", "thalach", "exang", "oldpeak", "slope",
               "ca", "thal", "target"]

SYSTEM_PROMPT = """You are a medical data generator. Given a patient profile and 
clinical constraints retrieved from medical guidelines, generate ONE synthetic 
patient record that is medically plausible and consistent.

You must respond with ONLY a valid JSON object using exactly these keys:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

Field definitions:
- age: integer (29-77)
- sex: 0=female, 1=male
- cp: chest pain type 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
- trestbps: resting blood pressure mmHg (90-200)
- chol: serum cholesterol mg/dL (100-600)
- fbs: fasting blood sugar >120 mg/dL (0=no, 1=yes)
- restecg: resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)
- thalach: max heart rate achieved (70-210)
- exang: exercise induced angina (0=no, 1=yes)
- oldpeak: ST depression (0.0-6.2)
- slope: slope of peak exercise ST (0=upsloping, 1=flat, 2=downsloping)
- ca: number of major vessels (0-3)
- thal: 1=normal, 2=fixed defect, 3=reversible defect
- target: 0=no disease, 1=disease

Respond with ONLY the JSON object, no explanation."""

async def get_constraints(rag, profile: dict) -> str:
    query = (
        f"Clinical constraints for: age={profile['age']}, "
        f"sex={'male' if profile['sex']==1 else 'female'}, "
        f"cholesterol={profile.get('chol', 'unknown')} mg/dL, "
        f"blood pressure={profile.get('trestbps', 'unknown')} mmHg. "
        f"What ranges and correlations apply for heart disease risk?"
    )
    result = await rag.aquery(query, param=QueryParam(mode="naive"))
    return result

def generate_record(profile: dict, constraints: str) -> dict:
    prompt = f"""Clinical constraints from guidelines:
{constraints}

Seed patient profile (use as starting point, adjust to be medically consistent):
{json.dumps(profile, indent=2)}

Generate ONE complete synthetic patient record as a JSON object."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.7, "num_ctx": 2048}
    )

    raw = response["message"]["content"].strip()
    # Strip markdown code fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    record = json.loads(raw)

    # Validate all expected keys are present
    for col in UCI_COLUMNS:
        if col not in record:
            raise ValueError(f"Missing field in generated record: {col}")
    return record

async def generate_one(profile: dict) -> dict:
    rag = await build_rag()
    constraints = await get_constraints(rag, profile)
    print(f"\n--- Retrieved Constraints (truncated) ---")
    print(constraints[:300], "...")
    record = generate_record(profile, constraints)
    await rag.finalize_storages()
    return record

if __name__ == "__main__":
    # Test with a seed profile
    seed = {"age": 55, "sex": 1, "chol": 240, "trestbps": 140, "fbs": 1}
    result = asyncio.run(generate_one(seed))
    print("\n--- Generated Record ---")
    print(json.dumps(result, indent=2))
