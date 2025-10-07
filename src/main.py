from src.data_loader import load_data
from src.preprocessing import clean_data, build_preprocessor
from src.modeling import train_model
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import joblib
import os

PATH = "data/loan_default_raw.csv"

df = load_data(PATH)
df_p = clean_data(df)

numeric_cols = ['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio','loan_to_income','Education_ord']
categorical_cols = ['EmploymentType','MaritalStatus','LoanPurpose']

X = df_p[numeric_cols + categorical_cols + ['HasMortgage','HasDependents','HasCoSigner']]
y = df_p['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = build_preprocessor(numeric_cols, categorical_cols)

best_model = train_model(preprocessor, X_train, y_train, model_name="rf")
metrics = evaluate_model(best_model, X_test, y_test)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_rf_pipeline.joblib")
print("✅ Модель сохранена в models/best_rf_pipeline.joblib")
