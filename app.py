
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Estimate the likelihood that a customer may stop "
    "using a service."
)

st.info(
    "Educational demo using synthetic data. "
    "Predictions are estimates, not guarantees."
)

with st.form("customer_form"):
    tenure = st.number_input(
        "Tenure (months)", 1, 72, 12
    )
    monthly_charges = st.number_input(
        "Monthly charges", 20.0, 120.0, 75.0
    )
    usage_hours = st.number_input(
        "Usage hours per month", 1, 160, 60
    )
    support_calls = st.number_input(
        "Support calls", 0, 8, 2
    )
    contract_type = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic", "Bank transfer", "Credit card"]
    )
    internet_service = st.selectbox(
        "Internet service",
        ["Fiber", "DSL", "None"]
    )

    submitted = st.form_submit_button(
        "Estimate Churn Risk"
    )

if submitted:
    customer = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "usage_hours": usage_hours,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service
    }])

    probability = model.predict_proba(customer)[0, 1]
    prediction = model.predict(customer)[0]

    st.subheader("Prediction")
    st.metric(
        "Estimated churn probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning(
            "The model flags this customer as higher churn risk."
        )
    else:
        st.success(
            "The model does not flag this customer as likely to churn."
        )

    st.caption(
        "This is a model estimate from synthetic training data, "
        "not a reliable real-world risk assessment. "
        "Do not use it alone to make decisions about customers."
    )
