# Customer Churn Prediction

A Streamlit web app that predicts whether a bank customer is likely to churn,
powered by a TensorFlow neural network trained on customer demographics and account data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## Overview

Given inputs like credit score, geography, age, balance, and activity status,
the model outputs a churn probability score and classifies the customer as
likely to churn or not.

## Features

- Interactive UI with sliders, dropdowns, and number inputs for all customer fields
- Real-time churn probability score with a visual risk progress bar
- Full preprocessing pipeline — one-hot encoding, label encoding, and standard scaling
- Trained Keras `.h5` model with serialized sklearn encoders via `pickle`

## Project structure
```
├── app.py                  # Streamlit application
├── Models/
│   ├── model.h5            # Trained Keras model
│   ├── scaler.pkl          # StandardScaler
│   ├── label_encoder.pkl   # LabelEncoder (gender)
│   └── onehot_encoder.pkl  # OneHotEncoder (geography)
└── requirements.txt
```

## Getting started
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input features

Credit score · Geography · Gender · Age · Tenure · Balance ·
Number of products · Has credit card · Is active member · Estimated salary