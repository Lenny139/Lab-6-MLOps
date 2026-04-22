from pathlib import Path

import pandas as pd


FEATURE_CANDIDATES = [
    "age",
    "overall",
    "potential",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "position",
    "player_positions",
    "nationality",
    "nationality_name",
]
TARGET_COLUMN = "value_eur"


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _resolve_input_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_archive_dirs = [
        repo_root / "archive",
        repo_root.parent / "archive",
    ]

    archive_dir = next((path for path in candidate_archive_dirs if path.exists()), None)
    if archive_dir is None:
        return []

    preferred_files = [
        archive_dir / "male_players (legacy).csv",
        archive_dir / "female_players (legacy).csv",
        archive_dir / "male_players.csv",
        archive_dir / "female_players.csv",
    ]
    return [file_path for file_path in preferred_files if file_path.exists()]


def build_dataset() -> Path:
    input_files = _resolve_input_files()
    if not input_files:
        raise FileNotFoundError("No se encontraron CSV de jugadores dentro de la carpeta archive.")

    print("Archivos fuente:", [str(path) for path in input_files])
    frames = []
    for file_path in input_files:
        print(f"Leyendo {file_path.name}...")
        frames.append(pd.read_csv(file_path, low_memory=False))

    df = pd.concat(frames, ignore_index=True)

    position_col = _first_existing_column(df, ["position", "player_positions"])
    nationality_col = _first_existing_column(df, ["nationality", "nationality_name"])

    required_cols = [
        "age",
        "overall",
        "potential",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
        TARGET_COLUMN,
    ]

    if position_col is not None:
        required_cols.append(position_col)
    if nationality_col is not None:
        required_cols.append(nationality_col)

    available_cols = [column for column in required_cols if column in df.columns]
    missing_required = [
        column
        for column in [
            "age",
            "overall",
            "potential",
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physic",
            TARGET_COLUMN,
        ]
        if column not in available_cols
    ]
    if missing_required:
        raise ValueError(f"Faltan columnas esenciales para entrenar: {missing_required}")

    df = df[available_cols].copy()

    if position_col and position_col != "position":
        df = df.rename(columns={position_col: "position"})
        position_col = "position"
    if nationality_col and nationality_col != "nationality":
        df = df.rename(columns={nationality_col: "nationality"})
        nationality_col = "nationality"

    before_dropna = len(df)
    df = df.dropna().drop_duplicates()
    print(f"Filas eliminadas por nulos/duplicados: {before_dropna - len(df)}")

    categorical_cols = [column for column in ["position", "nationality"] if column in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    feature_cols = [column for column in df.columns if column != TARGET_COLUMN]
    df = df[feature_cols + [TARGET_COLUMN]]

    output_path = Path(__file__).resolve().parent / "dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset listo en: {output_path}")
    print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
    print(f"Última columna (target): {df.columns[-1]}")

    return output_path


if __name__ == "__main__":
    build_dataset()