import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает и подготавливает данные"""
    edu_map = {"High School":1, "Bachelor's":2, "Master's":3, "PhD":4}
    empl_map = {"Part-time":1, "Unemployed":2, "Self-employed":3, "Full-time":4}
    mar_map = {"Married":1, "Divorced":2, "Single":3}
    purp_map = {"Business":1, "Home":2, "Education":3, "Auto":4, "Other":5}

    df_p = df.copy()

    if 'LoanID' in df_p.columns:
        df_p = df_p.drop(columns=['LoanID'])

    for c in ['HasMortgage','HasDependents','HasCoSigner']:
        if c in df_p.columns:
            df_p[c] = df_p[c].map({'Yes':1, 'No':0})

    df_p['Education_ord'] = df_p['Education'].map(edu_map)
    df_p['EmploymentType_ord'] = df_p['EmploymentType'].map(empl_map)
    df_p['MaritalStatus_ord'] = df_p['MaritalStatus'].map(mar_map)
    df_p['LoanPurpose_ord'] = df_p['LoanPurpose'].map(purp_map)

    if set(['LoanAmount','Income']).issubset(df_p.columns):
        df_p['loan_to_income'] = df_p['LoanAmount'] / (df_p['Income'].replace(0, np.nan))

    return df_p


def build_preprocessor(numeric_cols, categorical_cols):
    """Создаёт пайплайн препроцессинга"""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')

    return preprocessor