import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('Models/model.h5')

    with open('Models/onehot_encoder.pkl', 'rb') as f:
        onehot_geo = pickle.load(f)

    with open('Models/label_encoder.pkl', 'rb') as f:
        lb_gender = pickle.load(f)

    with open('Models/scaler.pkl', 'rb') as f:      
        scaler = pickle.load(f)

    return model, onehot_geo, lb_gender, scaler

try:
    model, onehot_geo, lb_gender, scaler = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model or encoders: {e}")
    st.stop()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔍")
st.title("🔍 Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict churn probability.")

# ── User inputs ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    geography      = st.selectbox('Geography',        onehot_geo.categories_[0])
    gender         = st.selectbox('Gender',           lb_gender.classes_)
    age            = st.slider('Age',                 18, 92, 35)
    tenure         = st.slider('Tenure (years)',      0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)

with col2:
    credit_score     = st.number_input('Credit Score',       min_value=300, max_value=900, value=650)
    balance          = st.number_input('Balance',            min_value=0.0, value=0.0, format="%.2f")
    estimated_salary = st.number_input('Estimated Salary',   min_value=0.0, value=50000.0, format="%.2f")  # fixed
    has_cr_card      = st.selectbox('Has Credit Card',       [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_active_member = st.selectbox('Is Active Member?',     [0, 1], format_func=lambda x: "Yes" if x else "No")

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict Churn", use_container_width=True, type="primary"):

    # Build base feature dict (fixed EstimatedSalary mapping)
    input_data = {
        'CreditScore':     credit_score,
        'Gender':          lb_gender.transform([gender])[0],
        'Age':             age,
        'Tenure':          tenure,
        'Balance':         balance,
        'NumOfProducts':   num_of_products,
        'HasCrCard':       has_cr_card,
        'IsActiveMember':  is_active_member,
        'EstimatedSalary': estimated_salary,   # fixed: was using wrong variable
    }

    input_df = pd.DataFrame([input_data])      # fixed: no double-wrapping

    # One-hot encode Geography and append
    geo_encoded    = onehot_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_geo.get_feature_names_out(['Geography'])
    )
    input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale and predict
    input_scaled      = scaler.transform(input_df)
    prediction_proba  = float(model.predict(input_scaled)[0][0])

    # ── Results display ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Prediction Result")

    col_a, col_b = st.columns(2)
    col_a.metric("Churn Probability", f"{prediction_proba:.1%}")

    if prediction_proba > 0.5:
        col_b.error("⚠️ Customer is **likely to churn**")
    else:
        col_b.success("✅ Customer is **not likely to churn**")

    st.progress(prediction_proba, text=f"Risk score: {prediction_proba:.1%}")