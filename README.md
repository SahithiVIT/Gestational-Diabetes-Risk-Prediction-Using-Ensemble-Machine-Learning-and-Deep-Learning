# 🤰 Gestational Diabetes Mellitus (GDM) Prediction

An intelligent **Machine Learning, Deep Learning, and Ensemble Learning system** designed to predict **Gestational Diabetes Mellitus (GDM)** using patient health and clinical features.

The project compares multiple classification algorithms, Deep Belief Network (DBN), and Stacking Ensemble techniques to evaluate their performance in GDM prediction.

---

## 📌 Project Overview

Gestational Diabetes Mellitus is a type of diabetes that can occur during pregnancy.

This project builds a predictive system that analyzes patient features such as:

* Age
* Number of Pregnancies
* Gestation
* BMI
* HDL
* Family History
* Unexplained Prenatal Loss
* Large Child History
* PCOS
* Systolic Blood Pressure
* Diastolic Blood Pressure
* OGTT
* Hemoglobin
* Sedentary Lifestyle
* Prediabetes

The system predicts whether a patient is likely to have **GDM or Non-GDM**.

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
Multiple ML Model Training
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
Patient GDM Prediction
```

---

## 🧠 Models Implemented

The following Machine Learning and Deep Learning models were compared:

| Model                  | Description                                 |
| ---------------------- | ------------------------------------------- |
| 🌲 Random Forest       | Ensemble of multiple decision trees         |
| 🌳 Extra Trees         | Highly randomized tree-based ensemble       |
| ⚡ XGBoost              | Optimized gradient boosting algorithm       |
| 💡 LightGBM            | Fast gradient boosting framework            |
| 🐱 CatBoost            | Advanced boosting algorithm                 |
| 📐 SVM                 | Support Vector Machine classifier           |
| 👥 KNN                 | K-Nearest Neighbors classifier              |
| 🌿 Decision Tree       | Tree-based classification model             |
| 📈 Gradient Boosting   | Sequential boosting algorithm               |
| 🔄 AdaBoost            | Adaptive boosting classifier                |
| 📊 Logistic Regression | Linear classification model                 |
| 🧠 MLP                 | Multi-Layer Perceptron Neural Network       |
| 🔗 DBN                 | Deep Belief Network using stacked RBMs      |
| 🏆 Stacking Ensemble   | Combination of RF, Extra Trees, and XGBoost |

---

## 🔗 Stacking Ensemble Architecture

```text
Random Forest ─────┐
                   │
Extra Trees ───────┼──► Logistic Regression ──► Final Prediction
                   │
XGBoost ───────────┘
```

The predictions from multiple base models are combined using **Logistic Regression as the meta-model**.

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

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Imbalanced-learn
* XGBoost
* LightGBM
* CatBoost

### Data Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* Google Colab

---

## ⚙️ Data Preprocessing

### 1️⃣ Missing Value Handling

Missing numerical values are handled using **Mean Imputation**.

```python
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
```

---

### 2️⃣ Feature Standardization

`StandardScaler` is used to bring features to a common scale.

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

---

### 3️⃣ Class Balancing Using SMOTE

SMOTE generates synthetic samples for the minority class.

```python
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
```

This reduces model bias towards the majority class.

---

### 4️⃣ Feature Selection

The top **10 informative features** are selected using the ANOVA F-test.

```python
selector = SelectKBest(score_func=f_classif, k=10)
X = selector.fit_transform(X, y)
```

---

## 📊 Model Evaluation Metrics

Models are evaluated using **5-Fold Cross Validation**.

The following metrics are used:

| Metric    | Purpose                              |
| --------- | ------------------------------------ |
| Accuracy  | Overall prediction correctness       |
| Precision | Accuracy of positive predictions     |
| Recall    | Ability to identify GDM patients     |
| F1 Score  | Balance between Precision and Recall |
| ROC-AUC   | Overall classification capability    |

For medical prediction, **Recall is especially important because missing a GDM patient can be more serious than a false positive**.

---

## 📈 Performance Comparison

The project generates comparison graphs for:

* 📊 Accuracy
* 🎯 Precision
* 🔍 Recall
* ⚖️ F1 Score
* 📈 ROC Curve

These visualizations help identify the best-performing model.

---

## 🔬 ROC Curve

The ROC curve evaluates the relationship between:

```text
True Positive Rate (TPR)
          VS
False Positive Rate (FPR)
```

A higher **Area Under the Curve (AUC)** indicates better model discrimination.

---

## 🩺 Patient Prediction

The system accepts patient clinical information as input.

```text
Age
Number of Pregnancies
Gestation
BMI
HDL
Family History
Prenatal Loss
Large Child History
PCOS
Systolic BP
Diastolic BP
OGTT
Hemoglobin
Sedentary Lifestyle
Prediabetes
```

The patient data undergoes the same preprocessing pipeline:

```text
Patient Input
     ↓
Missing Value Handling
     ↓
Standardization
     ↓
Feature Selection
     ↓
MLP Neural Network
     ↓
Prediction
```

### Possible Results

```text
Result: GDM Detected
```

or

```text
Result: No GDM
```

---

## 📂 Project Structure

```text
Gestational-Diabetes-Prediction/
│
├── Gestational_Diabetes_Prediction.ipynb
├── Gestational Diabetic Dat Set.xlsx
└── README.md
```

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Gestational-Diabetes-Prediction
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost lightgbm catboost openpyxl
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the Notebook

Open:

```text
Gestational_Diabetes_Prediction.ipynb
```

Run all cells sequentially.

---

## 💻 Run Using Google Colab

1. Open Google Colab.
2. Upload the `.ipynb` file.
3. Upload the dataset.
4. Install the required libraries.
5. Run all notebook cells.

---

## ✨ Key Features

* 🤖 Multiple Machine Learning Models
* 🧠 MLP Neural Network
* 🔗 Deep Belief Network
* 🏆 Stacking Ensemble Learning
* ⚖️ SMOTE Class Balancing
* 🎯 Feature Selection
* 🔄 5-Fold Cross Validation
* 📊 Model Performance Comparison
* 📈 ROC Curve Analysis
* 🩺 Real-Time Patient Input Prediction

---

## 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV or Bayesian Optimization
* SHAP or LIME explainability
* Larger multi-hospital dataset validation
* Streamlit-based interactive web application
* Cloud deployment
* Real-time clinical prediction dashboard

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The prediction generated by the model should **not be considered a medical diagnosis or a replacement for professional medical advice**.



