# Customer Churn Prediction

## 1. Project Overview
Customer Churn Prediction is a machine learning project that estimates whether a customer may stop using a service based on account details, usage patterns, and service information.

## 2. Objective
To build and evaluate a classification model that identifies potential customer churn using sample customer data.

## 3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 4. Dataset
The project uses 1,000 synthetically generated customer records.

### Input Features
- Tenure in months
- Monthly charges
- Usage hours per month
- Number of support calls
- Contract type
- Payment method
- Internet service type

### Target Variable
- `0` — No churn
- `1` — Churn

## 5. Machine Learning Algorithm
Random Forest Classifier

Categorical features are converted into numerical representations using One-Hot Encoding. The dataset is divided into 80% training data and 20% testing data.

## 6. Model Evaluation Results

Test samples: 200

- Accuracy: 80.5%
- Churn precision: 0.81
- Churn recall: 0.83
- Churn F1-score: 0.82
- ROC-AUC: 0.871

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 73 | 21 |
| Churn | 18 | 88 |

The model correctly classified 161 out of 200 test records.

## 7. Interactive Web Application
The Streamlit application accepts customer information and displays:
- Estimated churn probability
- A churn-risk prediction
- An explanation that predictions are estimates

## 8. How to Run the Project

Install dependencies:

`python -m pip install pandas numpy scikit-learn matplotlib streamlit joblib`

Generate the dataset:

`python create_dataset.py`

Train and evaluate the model:

`python train_model.py`

Launch the interactive application:

`python -m streamlit run app.py`

## 9. Project Files
- `create_dataset.py` — generates sample customer data
- `customer_churn.csv` — synthetic dataset
- `train_model.py` — trains and evaluates the model
- `churn_model.pkl` — saved trained model
- `confusion_matrix.png` — evaluation visualization
- `app.py` — interactive Streamlit application

## 10. Limitations
The dataset is synthetic and does not represent actual customer behavior. The model's evaluation results demonstrate the machine learning workflow, not its reliability on real business data. Predictions are estimates and should not be used alone to make customer-related decisions.

## 11. Conclusion
The Random Forest classifier achieved 80.5% accuracy and a ROC-AUC of 0.871 on the held-out synthetic test dataset. The project demonstrates data preparation, classification, model evaluation, and interactive prediction using Streamlit.

## Author
Jothi A