import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="GDM Risk Prediction",
    page_icon="🤰",
    layout="centered"
)

st.title("🤰 Gestational Diabetes Risk Prediction")
st.write("Enter patient clinical details to estimate GDM risk.")

TARGET = "Class Label(GDM /Non GDM)"


@st.cache_resource
def train_model():

    df = pd.read_excel("Gestational Diabetic Dat Set.xlsx")

    X_df = df.drop(["Case Number", TARGET], axis=1)
    y = df[TARGET]

    feature_names = X_df.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="mean")

    X_train = imputer.fit_transform(X_train)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=10
    )

    X_train = selector.fit_transform(
        X_train,
        y_train
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )

    model.fit(X_train, y_train)

    return (
        model,
        imputer,
        scaler,
        selector,
        feature_names
    )


model, imputer, scaler, selector, feature_names = train_model()


st.subheader("🩺 Patient Details")

with st.form("patient_form"):

    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=60.0,
        value=25.0
    )

    preg = st.number_input(
        "Number of Pregnancies",
        min_value=0.0,
        max_value=15.0,
        value=1.0
    )

    gest = st.number_input(
        "Gestation",
        min_value=0.0,
        value=20.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    hdl = st.number_input(
        "HDL",
        min_value=0.0,
        value=50.0
    )

    family = st.selectbox(
        "Family History",
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
        "Systolic BP",
        min_value=50.0,
        max_value=250.0,
        value=120.0
    )

    dia_bp = st.number_input(
        "Diastolic BP",
        min_value=30.0,
        max_value=150.0,
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

    # Create dataframe using exact training column order
    user_data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    user_data = imputer.transform(user_data)

    user_data = scaler.transform(user_data)

    user_data = selector.transform(user_data)

    probability = model.predict_proba(
        user_data
    )[0][1]

    st.subheader("📊 Prediction Result")

    if probability >= 0.5:

        st.error("⚠️ Higher GDM Risk")

    else:

        st.success("✅ Lower GDM Risk")

    st.write(
        f"Estimated GDM Risk Score: "
        f"**{probability * 100:.2f}%**"
    )

    st.progress(float(probability))

    st.info(
        "This is a model-generated risk estimate "
        "and not a medical diagnosis."
    )


st.divider()

st.caption(
    "⚠️ Educational and research purposes only. "
    "Consult a qualified healthcare professional "
    "for medical evaluation."
)
