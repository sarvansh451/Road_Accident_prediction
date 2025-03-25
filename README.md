# 🚀 Machine Learning Road Accident Prediction
This project is a **web application** that allows users to interact with a trained **Machine Learning model** for either **classification** or **regression** tasks. It includes data preprocessing, model training, evaluation, and deployment using **Flask/FastAPI** for the back-end and **HTML, CSS, JavaScript (or React)** for the front-end.  

---

## 📌 Project Overview  

- ✅ **Data Preprocessing** – Normalization, scaling, PCA, and cross-validation.  
- ✅ **Model Training & Evaluation** – Trained on multiple classifiers/regressors and optimized using GridSearchCV.  
- ✅ **Web Application** – A front-end interface for user interaction with the trained model.  

Users can input feature values, and the application will return predictions based on the trained model.  

---

## 📊 Dataset & Model Details  

### **For Classification**  
- Applied **6 different classifiers** and evaluated performance.  
- Preprocessing techniques:  
  - **L1 & L2 Normalization**  
  - **Min-Max Scaling**  
  - **Standard Scaling**  
- Used **GridSearchCV** and **10-fold Cross-Validation** for hyperparameter tuning.  
- **Applied PCA** to reduce dimensionality.  

### **For Regression**  
- Applied **6 different regressors** and evaluated performance.  
- Used **GridSearchCV** and **10-fold Cross-Validation** for hyperparameter tuning.  

---

## 🖥️ Web Application  

The web application allows users to:  
1. Enter input features through a web-based form.  
2. Submit the data to a **Flask/FastAPI** back-end.  
3. Receive and display the model’s prediction in real time.  

### **Tech Stack Used**  
- **Front-End:** HTML, CSS, JavaScript (or React)  
- **Back-End:** Flask / FastAPI (Python)  
- **Machine Learning:** Scikit-Learn, Pandas, NumPy  


