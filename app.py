import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE


st.set_page_config(
    page_title="GDM Risk Prediction",
    page_icon="🤰",
    layout="centered"
)

st.title("🤰 Gestational Diabetes Risk Prediction")

st.write(
    "Enter patient clinical details to estimate "
    "Gestational Diabetes Mellitus (GDM) risk."
)

TARGET = "Class Label(GDM /Non GDM)"


@st.cache_resource
def train_model():

    df = pd.read_excel(
        "Gestational Diabetic Dat Set.xlsx"
    )

    X = df.drop(
        ["Case Number", TARGET],
        axis=1
    )

    y = df[TARGET]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="mean")

    X_train = imputer.fit_transform(X_train)

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    )

    model = CalibratedClassifierCV(
        rf,
        method="sigmoid",
        cv=5
    )

    model.fit(X_train, y_train)

    return model, imputer, feature_names


model, imputer, feature_names = train_model()


st.subheader("🩺 Enter Patient Details")


with st.form("patient_form"):

    age = st.number_input(
        "Age",
        18.0,
        60.0,
        25.0
    )

    preg = st.number_input(
        "Number of Pregnancies",
        0.0,
        15.0,
        1.0
    )

    gest = st.number_input(
        "Gestation",
        0.0,
        50.0,
        20.0
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        22.0
    )

    hdl = st.number_input(
        "HDL",
        0.0,
        150.0,
        55.0
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
        50.0,
        250.0,
        110.0
    )

    dia_bp = st.number_input(
        "Diastolic Blood Pressure",
        30.0,
        150.0,
        70.0
    )

    ogtt = st.number_input(
        "OGTT",
        0.0,
        400.0,
        100.0
    )

    hemo = st.number_input(
        "Hemoglobin",
        0.0,
        25.0,
        12.0
    )

    sedentary = st.selectbox(
        "Sedentary Lifestyle",
        ["No", "Yes"]
    )

    prediabetes = st.selectbox(
        "Prediabetes",
        ["No", "Yes"]
    )

    predict = st.form_submit_button(
        "🔍 Predict GDM"
    )


if predict:

    values = [
        age,
        preg,
        gest,
        bmi,
        hdl,
        1 if family == "Yes" else 0,
        1 if unexplained == "Yes" else 0,
        1 if large_child == "Yes" else 0,
        1 if pcos == "Yes" else 0,
        sys_bp,
        dia_bp,
        ogtt,
        hemo,
        1 if sedentary == "Yes" else 0,
        1 if prediabetes == "Yes" else 0
    ]

    user_data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    user_data = imputer.transform(user_data)

    probabilities = model.predict_proba(user_data)[0]

    class_index = list(model.classes_).index(1)

    probability = probabilities[class_index]

    st.subheader("📊 Prediction Result")

    if probability >= 0.5:

        st.error("⚠️ Higher GDM Risk")

    else:

        st.success("✅ Lower GDM Risk")

    st.metric(
        "Estimated GDM Risk Score",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.info(
        "This is a model-generated risk estimate "
        "and not a medical diagnosis."
    )


st.divider()

st.caption(
    "⚠️ Educational and research purposes only. "
    "Consult a healthcare professional for medical evaluation."
)
