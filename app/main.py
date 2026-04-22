from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
import os

app = FastAPI(title="MLOps API")

MODEL_PATH = "/app/modelos/model.pkl"
NUMERIC_FEATURES = [
    "age",
    "overall",
    "potential",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]

# Función para intentar cargar el modelo con reintentos
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def preprocess_for_inference(df: pd.DataFrame, model) -> pd.DataFrame:
    data = df.copy()

    position_col = "position" if "position" in data.columns else "player_positions" if "player_positions" in data.columns else None
    nationality_col = "nationality" if "nationality" in data.columns else "nationality_name" if "nationality_name" in data.columns else None

    if position_col is not None and position_col != "position":
        data = data.rename(columns={position_col: "position"})
        position_col = "position"
    if nationality_col is not None and nationality_col != "nationality":
        data = data.rename(columns={nationality_col: "nationality"})
        nationality_col = "nationality"

    missing_numeric = [col for col in NUMERIC_FEATURES if col not in data.columns]
    if missing_numeric:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas numéricas requeridas: {missing_numeric}",
        )

    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[NUMERIC_FEATURES].isnull().any().any():
        raise HTTPException(
            status_code=400,
            detail="Hay valores nulos o no numéricos en columnas numéricas requeridas.",
        )

    categorical_present = [col for col in ["position", "nationality"] if col in data.columns]
    if categorical_present and data[categorical_present].isnull().any().any():
        raise HTTPException(
            status_code=400,
            detail="Hay valores nulos en columnas categóricas requeridas.",
        )

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        missing_features = [feature for feature in expected_features if feature not in data.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas para predecir: {missing_features}",
            )
        data = data[expected_features]

    return data

@app.get("/")
def health_check():
    model = get_model()
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict-csv")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no encontrado en el volumen.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df_processed = preprocess_for_inference(df, model)
        preds = model.predict(df_processed)
        result = df.copy()
        result["prediction"] = preds
        return result.to_dict(orient="records")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))