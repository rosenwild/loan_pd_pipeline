import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Загружает датасет и возвращает DataFrame"""
    df = pd.read_csv(path)
    print(f"✅ Данные загружены: {df.shape}")
    return df