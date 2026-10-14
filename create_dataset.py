
import pandas as pd
import numpy as np

# Generate reproducible illustrative customer data
rng = np.random.default_rng(42)
n = 1000

tenure = rng.integers(1, 73, n)
monthly_charges = rng.integers(20, 121, n).astype(float)
usage_hours = rng.integers(1, 161, n)
support_calls = rng.integers(0, 9, n)
contract_type = rng.choice(
    ["Month-to-month", "One year", "Two year"],
    size=n,
    p=[0.55, 0.25, 0.20]
)
payment_method = rng.choice(
    ["Electronic", "Bank transfer", "Credit card"],
    size=n
)
internet_service = rng.choice(
    ["Fiber", "DSL", "None"],
    size=n,
    p=[0.5, 0.4, 0.1]
)

# Create an illustrative churn probability
logit = (
    0.8
    - 0.045 * tenure
    + 0.025 * (monthly_charges - 70)
    - 0.012 * usage_hours
    + 0.35 * support_calls
    + 1.0 * (contract_type == "Month-to-month")
    - 0.8 * (contract_type == "Two year")
    + 0.5 * (internet_service == "Fiber")
)

probability = 1 / (1 + np.exp(-logit))
churn = rng.binomial(1, probability)

data = pd.DataFrame({
    "tenure_months": tenure,
    "monthly_charges": monthly_charges,
    "usage_hours": usage_hours,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "churn": churn
})

data.to_csv("customer_churn.csv", index=False)

print("CUSTOMER CHURN DATASET")
print("=" * 40)
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nChurn distribution:")
print(data["churn"].value_counts())
print("\nDataset saved as customer_churn.csv")
print("Note: This is synthetic sample data.")
