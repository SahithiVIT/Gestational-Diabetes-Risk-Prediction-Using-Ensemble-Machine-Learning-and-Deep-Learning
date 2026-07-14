import streamlit as st
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="GDM Prediction",
    page_icon="🤰",
    layout="centered"
)

st.title("🤰 Gestational Diabetes Prediction")
st.write(
    "Enter the patient's clinical details to predict "
    "Gestational Diabetes Mellitus (GDM)."
)


# ---------------- TRAIN MODEL ----------------

@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_excel(
        "Gestational Diabetic Dat Set.xlsx"
    )

    # Shuffle dataset
    df = shuffle(df, random_state=42)

    # Features
    X = df.drop(
        [
            "Case Number",
            "Class Label(GDM /Non GDM)"
        ],
        axis=1
    )

    # Target
    y = df["Class Label(GDM /Non GDM)"]

    # Missing value handling
    imputer = SimpleImputer(strategy="mean")

    X = imputer.fit_transform(X)

    # Standardization
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # SMOTE
    smote = SMOTE(random_state=42)

    X, y = smote.fit_resample(X, y)

    # Feature Selection
    selector = SelectKBest(
        score_func=f_classif,
        k=10
    )

    X = selector.fit_transform(X, y)

    # MLP Neural Network
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        random_state=42
    )

    # Train model
    model.fit(X, y)

    return model, imputer, scaler, selector


# Load trained model
model, imputer, scaler, selector = train_model()


# ---------------- PATIENT INPUT ----------------

st.subheader("🩺 Enter Patient Details")


with st.form("patient_form"):

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=25
    )

    preg = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=15,
        value=1
    )

    gest = st.number_input(
        "Gestation",
        min_value=0.0,
        value=20.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    hdl = st.number_input(
        "HDL",
        min_value=0.0,
        value=50.0
    )

    family = st.selectbox(
        "Family History of Diabetes",
        ["No", "Yes"]
    )

    unexplained = st.selectbox(
        "Unexplained Prenatal Loss",
        ["No", "Yes"]
    )

    large_child = st.selectbox(
        "Large Child History",
        ["No", "Yes"]
    )

    pcos = st.selectbox(
        "PCOS",
        ["No", "Yes"]
    )

    sys_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=0.0,
        value=120.0
    )

    dia_bp = st.number_input(
        "Diastolic Blood Pressure",
        min_value=0.0,
        value=80.0
    )

    ogtt = st.number_input(
        "OGTT",
        min_value=0.0,
        value=120.0
    )

    hemo = st.number_input(
        "Hemoglobin",
        min_value=0.0,
        value=12.0
    )

    sedentary = st.selectbox(
        "Sedentary Lifestyle",
        ["No", "Yes"]
    )

    prediabetes = st.selectbox(
        "Prediabetes",
        ["No", "Yes"]
    )


    predict_button = st.form_submit_button(
        "🔍 Predict GDM"
    )


# ---------------- PREDICTION ----------------

if predict_button:

    # Convert Yes / No to 1 / 0

    family = 1 if family == "Yes" else 0

    unexplained = (
        1 if unexplained == "Yes" else 0
    )

    large_child = (
        1 if large_child == "Yes" else 0
    )

    pcos = 1 if pcos == "Yes" else 0

    sedentary = (
        1 if sedentary == "Yes" else 0
    )

    prediabetes = (
        1 if prediabetes == "Yes" else 0
    )


    # Create patient data

    user_data = np.array([[
        age,
        preg,
        gest,
        bmi,
        hdl,
        family,
        unexplained,
        large_child,
        pcos,
        sys_bp,
        dia_bp,
        ogtt,
        hemo,
        sedentary,
        prediabetes
    ]])


    # Apply preprocessing

    user_data = imputer.transform(user_data)

    user_data = scaler.transform(user_data)

    user_data = selector.transform(user_data)


    # Prediction

    prediction = model.predict(user_data)

    probability = model.predict_proba(
        user_data
    )[0][1]


    # ---------------- RESULT ----------------

    st.subheader("📊 Prediction Result")


    if prediction[0] == 1:

        st.error(
            "⚠️ GDM Detected"
        )

        st.write(
            f"GDM Risk Probability: "
            f"**{probability * 100:.2f}%**"
        )

        st.warning(
            "Please consult a healthcare "
            "professional for further evaluation."
        )


    else:

        st.success(
            "✅ No GDM Detected"
        )

        st.write(
            f"GDM Risk Probability: "
            f"**{probability * 100:.2f}%**"
        )


# ---------------- DISCLAIMER ----------------

st.divider()

st.caption(
    "⚠️ Disclaimer: This application is developed "
    "for educational and research purposes only. "
    "It should not be considered a medical diagnosis."
)
