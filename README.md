# 🧑‍⚕️ Multi-Disease Prediction System

A Streamlit-based Machine Learning Web App that predicts the likelihood of Heart Disease, Parkinson's Disease, and Diabetes using trained ML models.

🔗 **Live Demo:** [Multi-Disease Prediction App](https://multi-disease-prediction-system-by-ritikgarg7879.streamlit.app/)

---

## 🚀 Features

### Three Diseases Supported
- ❤️ Heart Disease  
- 🧠 Parkinson’s Disease  
- 💉 Diabetes  

### Machine Learning Models
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Decision Tree  
- KNN  
- XGBoost  

### Model Selection & Tuning
- Performed EDA on datasets  
- Used RandomizedSearchCV and GridSearchCV for hyperparameter tuning  
- Applied Cross-Validation for robust evaluation  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Curve  
- Confusion Matrix  
- Feature Importance (for Heart Disease)  

---

## 📊 Datasets Used
- Diabetes Dataset: PIMA Indian Diabetes Dataset  
- Heart Disease Dataset: UCI Cleveland Heart Disease Dataset  
- Parkinson’s Disease Dataset: UCI Parkinson’s Dataset  

---

## 🧪 Workflow

1. **Exploratory Data Analysis (EDA)**
   - Checked missing values, distributions, correlations using pandas, matplotlib, seaborn.  
   - Plotted histograms, heatmaps, and feature importance.  

2. **Model Training**
   - Split datasets into train/test sets.  
   - Trained multiple models (Logistic Regression, RF, SVM, NB, KNN, XGBoost).  

3. **Model Evaluation**
   - Used Accuracy, Precision, Recall, F1-score.  
   - Visualized Confusion Matrices.  
   - Generated ROC Curves & AUC Scores.  

4. **Hyperparameter Tuning**  
   - RandomizedSearchCV & GridSearchCV used for optimal parameters.  
   - Example (Diabetes):  
     - Logistic Regression Best Params → `{ 'C': 0.233, 'solver': 'liblinear' }`  
     - Random Forest Best Params → `{ 'n_estimators': 210, 'max_depth': 3, 'min_samples_leaf': 19 }`  

5. **Cross-Validation**  
   - Performed 5-fold cross-validation for accuracy, precision, recall, F1.  
   - Average metrics:  
     - Accuracy ≈ 84–89%  
     - Precision ≈ 82%  
     - Recall ≈ 92%  
     - F1 Score ≈ 87%  

6. **Model Saving**  
   - Best models saved with Joblib (.pkl files).  
   - Example: `parkinsons_rf_model.pkl`.  

7. **Deployment**  
   - Built frontend with Streamlit.  
   - Deployed app on Streamlit Cloud.  

---

## 📈 Results Summary

### ✅ Heart Disease
- Logistic Regression (GridSearchCV tuned) → Accuracy ~ 89%  
- Feature importance visualized (Age, Sex, CP, Chol, Thalach etc.)  

### ✅ Diabetes
- Logistic Regression (best tuned) → Accuracy ~ 88%  
- Random Forest → Accuracy ~ 87%  
- KNN (best k=15) → 75%  

### ✅ Parkinson’s Disease
- Random Forest Classifier → Accuracy ~ 89-93%  
- Metrics:  
  - Accuracy: 0.898  
  - Precision: 0.954  
  - Recall: 0.923  
  - F1: 0.96  

---

## 🛠️ Tech Stack

- Frontend: Streamlit  
- Backend Models: Scikit-learn, XGBoost  
- Visualization: Matplotlib, Seaborn  
- Model Persistence: Joblib  
- Deployment: Streamlit Cloud  

---

## ⚡ How to Run Locally

Clone the Repository

git clone https://github.com/ritikgarg7879/Multi-Disease-Prediction-System.git
cd multi-disease-prediction

---

### Create Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate # For Linux/Mac
venv\Scripts\activate # For Windows


---

### Install Dependencies

pip install -r requirements.txt

Example `requirements.txt` contents:

streamlit==1.49.0
scikit-learn==1.5.2
joblib==1.4.2
streamlit-option-menu==0.4.0

---

### Run the App

streamlit run app.py



---

### Open in Browser

[http://localhost:8501](http://localhost:8501/)

---

## 🌐 Deployment on Streamlit Cloud

1. Push project to GitHub.  
2. Login to Streamlit Cloud.  
3. Select your repository → branch → `app.py`.  
4. Deploy 🚀  
5. Get public link (e.g., [https://multi-disease-prediction-system-by-ritikgarg7879.streamlit.app/](https://multi-disease-prediction-system-by-ritikgarg7879.streamlit.app/))  

---

## 📷 Screenshots

<img width="1915" height="929" alt="image" src="https://github.com/user-attachments/assets/b15ca144-a202-4705-ac9e-bcdbdaa14a94" />
<img width="1918" height="922" alt="image" src="https://github.com/user-attachments/assets/a3dd8fb6-4cec-459b-9796-34000342644e" />


---

## 👤 Author

**Ritik Garg**  
🎓 VIT Vellore  

🔗 [LinkedIn](https://www.linkedin.com/in/ritikgarg7879) | [GitHub](https://github.com/ritikgarg7879)

---

✨ This project demonstrates how Machine Learning + Streamlit can be used in healthcare prediction systems.

