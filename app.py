import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("### Predict student performance using Machine Learning")

st.sidebar.header("Project Information")
st.sidebar.write("**Model:** Random Forest Regressor")
st.sidebar.write("**Dataset:** Student Performance Dataset")

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    st.error("Model files not found.")
    st.stop()

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Student CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.success("Dataset uploaded successfully.")

    # Paste the new code here ↓↓↓

    # Convert categorical columns
    df_encoded = pd.get_dummies(df)

    # Match training features
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_names]

    if st.button("Predict Student Performance"):

        prediction = model.predict(df_encoded)

        result = df.copy()
        result["Predicted_G3"] = prediction
        st.subheader("📊 Predicted Grade Distribution")

        fig = px.histogram(
        result,
        x="Predicted_G3",
        nbins=10,
         title="Predicted Student Grades"
         )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction Results")
        st.dataframe(result)

        avg = prediction.mean()

        st.metric(
            "Average Predicted Grade",
            round(avg, 2)
        )

        if avg >= 15:
            st.success("🟢 Risk Level: Low")
        elif avg >= 10:
            st.warning("🟡 Risk Level: Medium")
        else:
            st.error("🔴 Risk Level: High")