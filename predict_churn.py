from pathlib import Path
from typing import List
from pydantic import BaseModel, field_validator
from joblib import load
import pandas as pd
import os

# Path to saved model artifact
MODEL_PATH = Path(__file__).resolve().parent / "churn_logistic_model.pkl"


class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # Coerce numeric-like strings to numbers (Pydantic v2 field validators)
    @field_validator('SeniorCitizen', mode='before')
    def to_int(cls, v):
        return int(v)

    @field_validator('tenure', mode='before')
    def tenure_to_int(cls, v):
        return int(v)

    @field_validator('MonthlyCharges', 'TotalCharges', mode='before')
    def to_float(cls, v):
        return float(v)

    # If using Pydantic v2, provide model_dump compatibility fallback in prepare_row_for_model


def load_artifact(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    artifact = load(path)
    # expected keys: model, feature_names, cat_cols
    model = artifact.get('model')
    feature_names = artifact.get('feature_names')
    cat_cols = artifact.get('cat_cols')
    if model is None or feature_names is None or cat_cols is None:
        raise ValueError('Model artifact missing required keys (model, feature_names, cat_cols)')
    return model, feature_names, cat_cols


def prepare_row_for_model(row: ChurnInput, feature_names: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Convert validated ChurnInput into a DataFrame with columns matching feature_names.
    Encode the observed category, then retain the training feature columns.
    Do not drop the first category for a single row: it would remove every
    categorical signal. Reindexing removes the reference categories instead.
    """
    # Build DataFrame from pydantic model (support model_dump for pydantic v2)
    try:
        d = row.model_dump()
    except Exception:
        d = row.dict()
    df = pd.DataFrame([d])

    # Ensure numeric columns
    for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Reindex to feature_names
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

    return df_encoded


def predict_from_dict(test_raw: dict, model, feature_names: List[str], cat_cols: List[str]):
    # Validate input
    validated = ChurnInput(**test_raw)
    X_row = prepare_row_for_model(validated, feature_names, cat_cols)

    # Check for NaNs
    if X_row.isnull().any().any():
        raise ValueError('NaNs detected in processed input; check numeric fields')

    proba = model.predict_proba(X_row)[:, 1][0]
    label = int(model.predict(X_row)[0])
    return label, float(proba)


if __name__ == '__main__':
    # Load model artifact
    model, feature_names, cat_cols = load_artifact(MODEL_PATH)

    # Example high-risk test input likely to churn
    test_raw_high_risk = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 99.90,
        'TotalCharges': 99.90
    }

    label, proba = predict_from_dict(test_raw_high_risk, model, feature_names, cat_cols)
    print('Input ->', test_raw_high_risk)
    print(f'Predicted label: {label} (1=Churn)')
    print(f'Predicted probability of churn: {proba:.4f}')
