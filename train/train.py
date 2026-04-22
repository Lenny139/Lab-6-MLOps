import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from prepare_dataset import build_dataset

def train():
    # 1. Cargar datos (si no existe dataset.csv, se genera automáticamente)
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        build_dataset()

    df = pd.read_csv(dataset_path)
    target_column = "value_eur"
    if target_column not in df.columns:
        raise ValueError(f"No se encontró la columna target '{target_column}' en dataset.csv")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = [
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
    categorical_features = [feature for feature in ["position", "nationality"] if feature in X.columns]

    missing_numeric = [feature for feature in numeric_features if feature not in X.columns]
    if missing_numeric:
        raise ValueError(f"Faltan columnas numéricas requeridas: {missing_numeric}")

    for feature in numeric_features:
        X[feature] = pd.to_numeric(X[feature], errors="coerce")
    X = X.dropna(subset=numeric_features)
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Entrenar
    print("Iniciando entrenamiento en el contenedor...")
    # Elegimos RandomForestRegressor porque captura relaciones no lineales,
    # es robusto con features heterogéneas y funciona muy bien como baseline en tabulares.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    algoritmo = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", algoritmo),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE en test: {rmse:,.2f}")

    # 3. GUARDAR EN EL VOLUMEN COMPARTIDO
    ruta_modelo = "/app/modelos/model.pkl"
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
    joblib.dump(model, ruta_modelo)
    print(f"✅ Modelo guardado exitosamente en {ruta_modelo}")

if __name__ == "__main__":
    train()