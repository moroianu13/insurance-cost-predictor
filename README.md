# 💰 Insurance Cost Predictor

A Streamlit web app that estimates medical insurance charges based on user input or uploaded CSV files. Built using a polynomial regression model trained with scikit-learn.

## 🚀 Features

- Predicts insurance costs from user input
- Batch prediction from uploaded CSV files
- Visual explanation of most influential features
- Prediction history tracking per session
- Clean UI with dark theme and emoji-enhanced interface

## 📊 Model

- Type: Polynomial Regression (degree=2)
- Trained on 2,700+ rows of health & insurance data
- Most impactful feature: Smoking status

## 📂 Files

- `insurance_app.py` – The Streamlit app
- `insurance_model.pkl` – The trained regression model
- `requirements.txt` – Python dependencies

## 📦 Run locally

```bash
pip install -r requirements.txt
streamlit run insurance_app.py
