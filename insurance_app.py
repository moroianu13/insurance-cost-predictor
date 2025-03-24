import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("insurance_model.pkl")

# ---- Page Config ----
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰", layout="centered")

# ---- Header Section ----
st.markdown("<h1 style='text-align: center; color: orange;'>💰 Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict your estimated medical insurance cost based on personal data.</p>", unsafe_allow_html=True)

# ---- Input Form ----
with st.form("user_input"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age", 18, 100, 30)
        sex = st.selectbox("🧍 Sex", ["Male", "Female"])
        bmi = st.slider("⚖️ BMI", 10.0, 50.0, 25.0)

    with col2:
        children = st.slider("👶 Number of Children", 0, 5, 0)
        smoker = st.selectbox("🚬 Smoker", ["Yes", "No"])
        region = st.selectbox("🌍 Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    submitted = st.form_submit_button("🔮 Predict")

# ---- Initialize Session History ----
if 'history' not in st.session_state:
    st.session_state.history = []

# ---- Prediction ----
if submitted:
    sex_val = 1 if sex == "Male" else 2
    smoker_val = 1 if smoker == "Yes" else 0
    region_map = {"Northeast": 1, "Northwest": 2, "Southeast": 3, "Southwest": 4}
    region_val = region_map[region]

    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    predicted_cost = model.predict(input_data)[0]

    st.success(f"💸 **Predicted Insurance Cost: ${predicted_cost:,.2f}**")

    # Optional tips
    if smoker_val:
        st.warning("🚬 Smoking is significantly increasing your insurance cost.")
    if bmi > 30:
        st.info("⚠️ Your BMI is in the obese range, which may raise your cost.")

    # Add to session history
    st.session_state.history.append({
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "Children": children,
        "Smoker": smoker,
        "Region": region,
        "Prediction": f"${predicted_cost:,.2f}"
    })

# ---- Prediction History Display ----
if st.checkbox("🕘 Show Prediction History"):
    st.write(pd.DataFrame(st.session_state.history))

# ---- Feature Importance Chart ----
st.markdown("This chart shows which features (or feature combinations) most influence the insurance cost prediction, based on the trained model's coefficients.")

if st.checkbox("📊 Show Feature Impact"):
    coefs = model.named_steps['model'].coef_
    feature_names = model.named_steps['poly'].get_feature_names_out(
        ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    )

    # Zip into a dict, sort by absolute coefficient value
    coef_series = dict(zip(feature_names, coefs))
    top_features = dict(sorted(coef_series.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

    # Format labels: ^2 → ² and space → ×
    labels = [f.replace('^2', '²').replace(' ', ' × ') for f in top_features.keys()]
    values = list(top_features.values())

    fig, ax = plt.subplots()
    ax.barh(labels, values)
    ax.set_title("Top 10 Most Influential Features")
    ax.invert_yaxis()
    st.pyplot(fig)


# ---- CSV Upload Prediction ----
st.markdown("---")
st.header("📁 Predict from CSV file")
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

if csv_file is not None:
    df_upload = pd.read_csv(csv_file)
    try:
        predictions = model.predict(df_upload)
        df_upload['predicted_cost'] = predictions
        st.write("✅ Predictions completed:")
        st.dataframe(df_upload)

        # Downloadable result
        csv_out = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# ---- About Section ----
st.markdown("---")
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This is a personal portfolio project created by **Adrian**.

    - 🔍 Predicts **medical insurance charges** based on user attributes
    - 🧠 Built using **Scikit-Learn (Polynomial Regression)**  
    - ⚡ Web UI powered by **Streamlit**
    - 💾 Supports both manual input and batch predictions via CSV
    - 📊 Shows feature impact and saves prediction history per session

    Made with ❤️ and metal \m/ in 2025.
    """)

# ---- Footer ----
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Made with ❤️ by Adrian • 2025</p>", unsafe_allow_html=True)
