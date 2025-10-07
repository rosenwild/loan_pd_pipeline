import pandas as pd
from src.preprocess import clean_data

def test_clean_data_removes_missing_values():
    """Проверяем, что функция удаляет строки с пропущенными значениями"""
    df = pd.DataFrame({
        "age": [25, None, 35],
        "income": [1000, 2000, None],
        "target": [0, 1, 0]
    })
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0, "Пропуски не были удалены"

def test_clean_data_keeps_target_column():
    """Проверяем, что целевая переменная не потерялась"""
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0]
    })
    cleaned = clean_data(df)
    assert "target" in cleaned.columns, "Целевая переменная потерялась"

def test_clean_data_returns_dataframe():
    """Проверяем, что возвращается DataFrame"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cleaned = clean_data(df)
    assert isinstance(cleaned, pd.DataFrame), "Функция должна возвращать DataFrame"