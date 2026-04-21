import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from prepare_dataset import build_dataset

def train():
    # 1. Cargar datos (si no existe dataset.csv, se genera automáticamente)
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        build_dataset()

    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Entrenar
    print("Iniciando entrenamiento en el contenedor...")
    # Elegimos RandomForestRegressor porque captura relaciones no lineales,
    # es robusto con features heterogéneas y funciona muy bien como baseline en tabulares.
    algoritmo = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model = algoritmo.fit(X_train, y_train)

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