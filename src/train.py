from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_model(preprocessor, X_train, y_train, model_name="rf", params=None):
    """Обучает модель с препроцессором"""
    if params is None:
        params = {}

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=42, **params)
    elif model_name == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif model_name == "xgb":
        clf = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            **params
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    pipe.fit(X_train, y_train)
    return pipe