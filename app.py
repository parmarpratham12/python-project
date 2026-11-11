import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Health Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(110deg, #124e78, #168a8a);
        box-shadow: 0 10px 28px rgba(18, 78, 120, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
    }
    .hero p {
        margin: 0.55rem 0 0;
        font-size: 1.05rem;
        opacity: 0.92;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h2,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    .stMarkdown,
    .stCaption {
        color: inherit;
    }
    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border: 0;
        border-radius: 12px;
        color: white;
        background: linear-gradient(90deg, #127c83, #176b9b);
        font-size: 1.05rem;
        font-weight: 700;
    }
    div.stButton > button:hover {
        color: white;
        border: 0;
        background: linear-gradient(90deg, #0d6970, #125b87);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the saved preprocessing artifacts
model_files = {
    "K-Nearest Neighbors (KNN)": "KNN_heart.pkl",
    "Decision Tree": "Decision Tree_heart.pkl",
    "Logistic Regression": "Logistic Regression_heart.pkl",
    "Naive Bayes": "Naive Bayes_heart.pkl",
    "Support Vector Machine (RBF)": "SVM (RBF Kernel)_heart.pkl",
}

scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.markdown(
    """
    <div class="hero">
        <h1>Heart Health Predictor</h1>
        <p>Enter your health details and choose a model to get a prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Choose your prediction model")
    selected_model = st.selectbox("Model", list(model_files))

with st.container(border=True):
    st.subheader("Your health details")
    left, middle, right = st.columns(3)

    with left:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

    with middle:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)

    with right:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    model = joblib.load(model_files[selected_model])
    prediction = model.predict(scaled_input)[0]

    # Show result
    st.caption(f"Prediction made with: {selected_model}")
    if prediction == 1:
        st.error("The model indicates a higher risk of heart disease.")
    else:
        st.success("The model indicates a lower risk of heart disease.")

st.caption("This tool is for informational purposes only and is not a medical diagnosis.")