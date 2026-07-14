# 🤰 Gestational Diabetes Risk Prediction Using Machine Learning and Deep Learning

An end-to-end **Machine Learning, Deep Learning, and Ensemble Learning project** for predicting the risk of **Gestational Diabetes Mellitus (GDM)** using patient clinical and health-related features.

The project compares multiple classification models, a Deep Belief Network (DBN), and a Stacking Ensemble using a leakage-aware machine learning pipeline.

An interactive Streamlit web application is also deployed for real-time GDM risk estimation.

---

## 🌐 Live Application

🚀 **Try the GDM Risk Prediction App**

https://bq6ncgt7rr47yvnyjc8pxn.streamlit.app/

The user can enter patient clinical details and receive an estimated GDM risk score.

---

## 📌 Project Overview

Gestational Diabetes Mellitus is a condition involving elevated blood glucose levels during pregnancy.

This project analyzes patient clinical features including:

- Age
- Number of Pregnancies
- Gestation
- BMI
- HDL
- Family History
- Unexplained Prenatal Loss
- Large Child History
- PCOS
- Systolic Blood Pressure
- Diastolic Blood Pressure
- OGTT
- Hemoglobin
- Sedentary Lifestyle
- Prediabetes

The system estimates whether the patient has:

- ⚠️ Higher GDM Risk
- ✅ Lower GDM Risk

---

## 🚀 Project Workflow

```text
Dataset
   ↓
Data Shuffling
   ↓
Feature and Target Separation
   ↓
5-Fold Cross Validation
   ↓
Imputation
   ↓
Standardization
   ↓
SMOTE Class Balancing
   ↓
ANOVA Feature Selection
   ↓
Machine Learning Models
   ↓
Deep Learning Models
   ↓
Stacking Ensemble
   ↓
Performance Evaluation
   ↓
ROC-AUC Analysis
   ↓
Probability Calibration
   ↓
GDM Risk Prediction
   ↓
Streamlit Deployment
```

---

## 🔄 Leakage-Aware Machine Learning Pipeline

The project uses an `imblearn` pipeline to apply preprocessing separately within each cross-validation training fold.

```text
Training Fold
     ↓
Mean Imputation
     ↓
StandardScaler
     ↓
SMOTE
     ↓
SelectKBest
     ↓
Model Training
     ↓
Validation Fold Evaluation
```

This approach helps reduce **data leakage** during cross-validation.

---

## 🧠 Models Implemented

| Model | Description |
|---|---|
| Random Forest | Ensemble of multiple decision trees |
| Extra Trees | Highly randomized tree ensemble |
| XGBoost | Optimized gradient boosting |
| LightGBM | Efficient gradient boosting framework |
| CatBoost | Advanced boosting algorithm |
| SVM | Support Vector Machine classifier |
| KNN | K-Nearest Neighbors |
| Decision Tree | Tree-based classifier |
| Gradient Boosting | Sequential boosting model |
| AdaBoost | Adaptive boosting classifier |
| Logistic Regression | Linear probabilistic classifier |
| MLP | Multi-Layer Perceptron Neural Network |
| DBN | Deep Belief Network using stacked RBMs |
| Stacking Ensemble | RF, Extra Trees, and XGBoost ensemble |

---

## 🔗 Stacking Ensemble Architecture

```text
Random Forest ─────┐
                   │
Extra Trees ───────┼──► Logistic Regression ──► Final Prediction
                   │
XGBoost ───────────┘
```

Random Forest, Extra Trees, and XGBoost are used as base models.

Logistic Regression is used as the meta-model to combine their predictions.

---

## 🧠 MLP Neural Network Architecture

```text
Input Features
      ↓
Hidden Layer - 128 Neurons
      ↓
Hidden Layer - 64 Neurons
      ↓
Hidden Layer - 32 Neurons
      ↓
Output Prediction
```

The MLP model is used to learn complex nonlinear relationships between patient clinical features and GDM risk.

Early stopping is used to help reduce overfitting.

---

## 🧠 Deep Belief Network Architecture

```text
Input Features
      ↓
RBM Layer - 128 Components
      ↓
RBM Layer - 64 Components
      ↓
RBM Layer - 32 Components
      ↓
Logistic Regression
      ↓
GDM / Non-GDM
```

The Deep Belief Network is implemented using stacked Restricted Boltzmann Machines followed by Logistic Regression.

---

## ⚙️ Data Preprocessing

### 1. Missing Value Handling

Missing numerical values are handled using Mean Imputation.

```python
SimpleImputer(strategy="mean")
```

---

### 2. Feature Standardization

`StandardScaler` transforms numerical features to a common scale.

```python
StandardScaler()
```

This is useful for models such as:

- SVM
- KNN
- Logistic Regression
- MLP

---

### 3. Class Balancing Using SMOTE

SMOTE is used to handle class imbalance.

```python
SMOTE(random_state=42)
```

SMOTE generates synthetic minority-class samples instead of simply duplicating existing records.

In the model evaluation pipeline, SMOTE is applied to the training fold.

---

### 4. Feature Selection

The top 10 informative features are selected using `SelectKBest`.

```python
SelectKBest(
    score_func=f_classif,
    k=10
)
```

The ANOVA F-test is used to measure the relationship between each feature and the target class.

---

## 📊 Model Evaluation

The models are evaluated using **5-Fold Cross Validation**.

The following metrics are compared:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify positive GDM cases |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between classes |

Recall is an important metric in risk-screening applications because missed positive cases may require further clinical attention.

---

## 📈 Performance Comparison

The project generates model comparison visualizations for:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

A ROC curve is also generated for the Stacking Ensemble model.

---

## 📈 ROC Curve

The ROC curve compares:

```text
True Positive Rate
        VS
False Positive Rate
```

The Area Under the Curve (AUC) measures the model's ability to distinguish between GDM and Non-GDM classes.

A higher AUC generally indicates better class discrimination.

---

## 🎯 Probability Calibration

The final MLP prediction pipeline uses probability calibration.

```python
CalibratedClassifierCV(
    final_mlp_pipeline,
    method="sigmoid",
    cv=5
)
```

Probability calibration is used to obtain more stable model probability estimates.

The final output is displayed as an estimated GDM risk score.

Example:

```text
Estimated GDM Risk Score: 24.50%

Result: Lower GDM Risk
```

---

## 🌐 Interactive Streamlit Application

The project includes an interactive Streamlit web application.

### Application Flow

```text
User Opens Web Application
          ↓
Enters Patient Details
          ↓
Clicks Predict GDM
          ↓
Data Preprocessing
          ↓
Trained Prediction Model
          ↓
Probability Estimation
          ↓
GDM Risk Score
          ↓
Higher / Lower GDM Risk
```

### Live Application

https://bq6ncgt7rr47yvnyjc8pxn.streamlit.app/

---

## ✨ Key Features

- Multiple Machine Learning Model Comparison
- Deep Learning using MLP
- Deep Belief Network using RBMs
- Stacking Ensemble Learning
- Leakage-Aware Preprocessing Pipeline
- SMOTE Class Balancing
- ANOVA-Based Feature Selection
- 5-Fold Cross Validation
- Multiple Evaluation Metrics
- ROC-AUC Analysis
- Probability Calibration
- Interactive Patient Input
- GDM Risk Score Estimation
- Streamlit Web Deployment

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning and Deep Learning

- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- CatBoost

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib

### Web Application

- Streamlit

### Development Tools

- Google Colab
- Jupyter Notebook
- GitHub

---

## 📂 Project Structure

```text
Gestational-Diabetes-Risk-Prediction/
│
├── app.py
├── requirements.txt
├── Gestational Diabetic Dat Set.xlsx
├── Gestational_Diabetes_Risk_Prediction.ipynb
└── README.md
```

---

## ▶️ Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/SahithiVIT/Gestational-Diabetes-Risk-Prediction-Using-Ensemble-Machine-Learning-and-Deep-Learning.git
```

### 2. Open the Project Directory

```bash
cd Gestational-Diabetes-Risk-Prediction-Using-Ensemble-Machine-Learning-and-Deep-Learning
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The interactive application is deployed using Streamlit Community Cloud.

### GitHub Repository

https://github.com/SahithiVIT/Gestational-Diabetes-Risk-Prediction-Using-Ensemble-Machine-Learning-and-Deep-Learning

### Live Application

https://bq6ncgt7rr47yvnyjc8pxn.streamlit.app/

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV or Bayesian Optimization
- SHAP or LIME model explainability
- External validation using larger datasets
- Multi-hospital dataset validation
- Model monitoring and versioning
- Improved clinical risk dashboard
- Additional patient risk insights

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The generated risk score is a model-based estimate and should **not be considered a medical diagnosis or a replacement for professional medical advice**.

