
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# 1. Load sample customer data
data = pd.read_csv("customer_churn.csv")

X = data.drop(columns=["churn"])
y = data["churn"]

numeric_features = [
    "tenure_months",
    "monthly_charges",
    "usage_hours",
    "support_calls"
]

categorical_features = [
    "contract_type",
    "payment_method",
    "internet_service"
]

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Convert categorical columns into numerical features
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features),
    ("numeric", "passthrough", numeric_features)
])

# 4. Build the classification pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate on unseen test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("CUSTOMER CHURN PREDICTION")
print("=" * 45)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 3
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 3
))

# 7. Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)
display.plot()
plt.title("Customer Churn Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 8. Save model and feature information
joblib.dump(model, "churn_model.pkl")

print("\nModel saved as churn_model.pkl")
print("Confusion matrix saved as confusion_matrix.png")
