# 🤰 Gestational Diabetes Risk Prediction Using Machine Learning and Deep Learning

An interactive Machine Learning, Deep Learning, and Ensemble Learning project designed to predict the risk of **Gestational Diabetes Mellitus (GDM)** using patient clinical and health-related features.

The project compares multiple classification models and provides a deployed Streamlit web application for interactive patient risk prediction.

---

## 🌐 Live Application

🚀 **Try the deployed GDM Prediction App:**

https://bq6ncgt7rr47yvnyjc8pxn.streamlit.app/

Users can enter patient clinical details and receive an instant GDM risk prediction.

---

## 📌 Project Overview

Gestational Diabetes Mellitus is a condition that can occur during pregnancy and is associated with increased blood glucose levels.

This project analyzes clinical features such as:

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

The system predicts:

- ⚠️ GDM Detected
- ✅ No GDM Detected

---

## 🚀 Project Workflow

```text
Dataset
   ↓
Data Shuffling
   ↓
Feature and Target Separation
   ↓
Missing Value Imputation
   ↓
Feature Standardization
   ↓
Class Distribution Analysis
   ↓
SMOTE Class Balancing
   ↓
Feature Selection
   ↓
Machine Learning Model Training
   ↓
Deep Learning Models
   ↓
Stacking Ensemble
   ↓
5-Fold Cross Validation
   ↓
Performance Comparison
   ↓
ROC Curve Analysis
   ↓
Interactive Patient Prediction
   ↓
Streamlit Deployment
```

---

## 🧠 Models Implemented

| Model | Description |
|---|---|
| Random Forest | Ensemble of multiple decision trees |
| Extra Trees | Randomized tree-based ensemble |
| XGBoost | Optimized gradient boosting algorithm |
| LightGBM | Fast gradient boosting framework |
| CatBoost | Advanced boosting algorithm |
| SVM | Support Vector Machine classifier |
| KNN | K-Nearest Neighbors |
| Decision Tree | Tree-based classification |
| Gradient Boosting | Sequential boosting model |
| AdaBoost | Adaptive boosting classifier |
| Logistic Regression | Linear classification model |
| MLP | Multi-Layer Perceptron Neural Network |
| DBN | Deep Belief Network using stacked RBMs |
| Stacking Ensemble | RF, Extra Trees and XGBoost ensemble |

---

## 🔗 Stacking Ensemble Architecture

```text
Random Forest ─────┐
                   │
Extra Trees ───────┼──► Logistic Regression ──► Final Prediction
                   │
XGBoost ───────────┘
```

The base model predictions are combined using Logistic Regression as the meta-model.

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

---

## ⚙️ Data Preprocessing

### Missing Value Handling

Missing numerical values are handled using Mean Imputation.

```python
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
```

### Feature Standardization

StandardScaler is used to bring numerical features to a common scale.

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Class Balancing Using SMOTE

SMOTE generates synthetic minority-class samples to reduce class imbalance.

```python
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
```

### Feature Selection

The top 10 informative features are selected using the ANOVA F-test.

```python
selector = SelectKBest(score_func=f_classif, k=10)
X = selector.fit_transform(X, y)
```

---

## 📊 Model Evaluation

Models are evaluated using 5-Fold Cross Validation.

The evaluation metrics include:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Accuracy of positive predictions |
| Recall | Ability to identify GDM cases |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Overall classification discrimination |

Recall is particularly important in risk-screening applications because missed positive cases may require further attention.

---

## 📈 Performance Analysis

The project compares models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- ROC Curve

These metrics help compare the predictive performance of different models.

---

## 🌐 Interactive Streamlit Web Application

The trained prediction pipeline is integrated into an interactive Streamlit application.

### Application Flow

```text
User Opens Web Application
          ↓
Enters Patient Details
          ↓
Clicks "Predict GDM"
          ↓
Data Preprocessing
          ↓
MLP Neural Network
          ↓
GDM Risk Prediction
          ↓
Prediction Result Displayed
```

### Live App

https://bq6ncgt7rr47yvnyjc8pxn.streamlit.app/

---

## ✨ Key Features

- Multiple Machine Learning Model Comparison
- MLP Neural Network
- Deep Belief Network
- Stacking Ensemble Learning
- SMOTE Class Balancing
- ANOVA-Based Feature Selection
- 5-Fold Cross Validation
- ROC-AUC Evaluation
- Interactive Patient Input
- GDM Risk Probability
- Real-Time Prediction
- Streamlit Web Deployment

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

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
- Seaborn

### Web Application

- Streamlit

### Development Tools

- Jupyter Notebook
- Google Colab
- GitHub

---

## 📂 Project Structure

```text
Gestational-Diabetes-Risk-Prediction/
│
├── app.py
├── requirements.txt
├── Gestational Diabetic Dat Set.xlsx
├── Yet_another_copy_of_GDM (1).ipynb
└── README.md
```

---

## ▶️ Run Locally

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

The application will open in your browser.

---

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

**GitHub Repository:**

https://github.com/SahithiVIT/Gestational-Diabetes-Risk-Prediction-Using-Ensemble-Machine-Learning-and-Deep-Learning

**Live Application:**

https://bq6ncgt7rr47yvnyjc8pxn.streamlit.app/

---

## 🔮 Future Improvements

- Hyperparameter tuning
- SHAP or LIME model explainability
- Validation using larger multi-hospital datasets
- Model versioning and monitoring
- Improved clinical dashboard
- Additional patient risk insights

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only.

The prediction generated by the application should not be considered a medical diagnosis or a replacement for professional medical advice.

---

## 👩‍💻 Author

**Sahithi Pilla**

Computer Science and Engineering  
VIT-AP University

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ Star.

Contributions, suggestions, and feedback are welcome.
