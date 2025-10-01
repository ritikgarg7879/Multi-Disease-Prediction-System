# 🧑‍⚕️ Multi-Disease Prediction System

A **Streamlit-based Machine Learning Web App** that predicts the likelihood of **Heart Disease**, **Parkinson's Disease**, and **Diabetes** using trained ML models.

🔗 **Live Demo:** [Multi-Disease Prediction App](https://multi-disease-prediction-system-by-ritikgarg7879.streamlit.app/)

---

## 🚀 Features

* Predicts **Heart Disease**, **Parkinson's Disease**, and **Diabetes**.
* Multiple ML models supported: Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Tree, KNN, XGBoost.
* Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`.
* Model evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
* Model persistence with `joblib` and easy deployment on Streamlit Cloud.

---

## 📊 Datasets

* **Diabetes**: PIMA Indian Diabetes Dataset.
* **Heart Disease**: UCI Cleveland Heart Disease Dataset.
* **Parkinson's Disease**: UCI Parkinson's Dataset.

---

## 🧪 Workflow

1. **EDA** — check missing values, distributions, correlations; plot histograms and heatmaps.
2. **Preprocessing** — scale/normalize features, handle missing values, encode categorical features.
3. **Train** — split into train/test, train multiple models, cross-validation.
4. **Hyperparameter Tuning** — use `RandomizedSearchCV` / `GridSearchCV` to find best parameters.
5. **Evaluate** — metrics, confusion matrices, ROC curves.
6. **Save Models** — persist best models using `joblib` (e.g., `parkinsons_rf_model.pkl`).
7. **Deploy** — build UI with Streamlit and deploy on Streamlit Cloud.

---

## 📈 Results Summary

* **Heart Disease**: Logistic Regression → Accuracy ≈ 89%
* **Diabetes**: Naive Bayes → Accuracy ≈ 88%
* **Parkinson's Disease**: Random Forest → Accuracy ≈ 89–93%

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Models:** scikit-learn, XGBoost
* **Visualization:** matplotlib, seaborn
* **Persistence:** joblib
* **Deployment:** Streamlit Cloud

---

## ⚡ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ritikgarg7879/Multi-Disease-Prediction-System.git
cd Multi-Disease-Prediction-System
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`**

```
streamlit
numpy
pandas
scikit-learn
xgboost
joblib
matplotlib
seaborn
```

### 4. Add Saved Models

Place `.pkl` files inside a `models/` folder:

* `models/diabetes_best_model.pkl`
* `models/heart_best_model.pkl`
* `models/parkinsons_rf_model.pkl`

### 5. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧩 Notes on Model Input Order

* **Diabetes (PIMA):** `[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]`
* **Heart (Cleveland):** columns in order used during training.
* **Parkinson's:** use acoustic features in same order as training.

> If pipelines were used, input raw values and preprocessing is handled automatically.

---

## ✅ Deploying to Streamlit Cloud

1. Push repository to GitHub.
2. Login to Streamlit Cloud and create new app.
3. Connect repo, select branch and `app.py`.
4. Deploy and share public link.

---

## 📷 Screenshots
<img width="1915" height="929" alt="Screenshot 2025-10-02 002026" src="https://github.com/user-attachments/assets/e75cc603-da94-40e4-8806-8e19a151628e" />
<img width="1914" height="910" alt="Screenshot 2025-10-02 002446" src="https://github.com/user-attachments/assets/ea155509-18bc-439c-957f-a8ff19bc8e84" />
<img width="1918" height="922" alt="Screenshot 2025-10-02 002846" src="https://github.com/user-attachments/assets/615f6afb-500a-4c59-a78d-b462adf0b201" />

---

## 👤 Author

**Ritik Garg** — VIT Vellore
GitHub: [ritikgarg7879](https://github.com/ritikgarg7879)
LinkedIn: [ritikgarg7879](www.linkedin.com/in/ritik-garg7879)

---
