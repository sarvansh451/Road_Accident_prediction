from flask import Flask, render_template, request, jsonify
import os, threading
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Specify template folder
current_dir = os.getcwd()
template_dir = os.path.join(current_dir, "templates")
app = Flask(__name__, template_folder=template_dir)

# Load dataset
# (Update the path to your CSV file accordingly)
class_data = pd.read_csv("C:/Users/KIIT/Documents/jupyter notebook/accident.csv")
class_data = pd.get_dummies(class_data, drop_first=True)
X_class = class_data.iloc[:, :-1].values  
y_class = class_data.iloc[:, -1].values
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# Define classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(solver='lbfgs', max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVC": SVC(probability=True),
    "GradientBoosting": GradientBoostingClassifier()
}

# Evaluate function (used for tables)
def evaluate_classifiers(X, y, classifiers):
    results = []
    for name, clf in classifiers.items():
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        results.append([name, acc, prec, rec, f1])
    return pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

# Table 1: Baseline classifier performance
table1 = evaluate_classifiers(X_train_class, y_train_class, classifiers)

# Table 2: Performance after scaling with various scalers
scalers = {
    "L1": Normalizer(norm='l1'),
    "L2": Normalizer(norm='l2'),
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler()
}
results = []
for scale_name, scaler in scalers.items():
    for name, clf in classifiers.items():
        pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])
        pipeline.fit(X_train_class, y_train_class)
        y_pred = pipeline.predict(X_train_class)
        acc = accuracy_score(y_train_class, y_pred)
        prec = precision_score(y_train_class, y_pred, zero_division=0)
        rec = recall_score(y_train_class, y_pred, zero_division=0)
        f1 = f1_score(y_train_class, y_pred, zero_division=0)
        results.append([scale_name, name, acc, prec, rec, f1])
table2 = pd.DataFrame(results, columns=["Scaler", "Model", "Accuracy", "Precision", "Recall", "F1-Score"])

# Table 3: GridSearchCV hyperparameter tuning for each classifier
param_grids = {
    "LogisticRegression": {"classifier__C": [0.1, 1, 10]},
    "KNN": {"classifier__n_neighbors": [3, 5, 7]},
    "DecisionTree": {"classifier__max_depth": [None, 5, 10]},
    "RandomForest": {"classifier__n_estimators": [50, 100]},
    "SVC": {"classifier__C": [0.1, 1, 10]},
    "GradientBoosting": {"classifier__n_estimators": [50, 100]}
}
grid_results = []
best_models = {}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for name, clf in classifiers.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    param_grid = param_grids.get(name, {})
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_class, y_train_class)
    best_score = grid.best_score_
    best_params = grid.best_params_
    grid_results.append([name, best_score, best_params])
    best_models[name] = grid.best_estimator_
table3 = pd.DataFrame(grid_results, columns=["Model", "Best CV Score", "Best Parameters"])

# Table 4: Performance after PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_class)
table4 = evaluate_classifiers(X_pca, y_train_class, classifiers)

# Home route to display evaluation tables
@app.route('/')
def index():
    return render_template('index.html', 
                           table1=table1.to_html(classes='table table-striped', index=False),
                           table2=table2.to_html(classes='table table-striped', index=False),
                           table3=table3.to_html(classes='table table-striped', index=False),
                           table4=table4.to_html(classes='table table-striped', index=False))

# API endpoint to get tables in JSON format (if needed)
@app.route('/get_tables')
def get_tables():
    return jsonify({
        "table1": table1.to_dict(orient="records"),
        "table2": table2.to_dict(orient="records"),
        "table3": table3.to_dict(orient="records"),
        "table4": table4.to_dict(orient="records")
    })

# Prediction route with GET for form and POST for predictions
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Expected input: comma separated values for features and model selection
        features_str = request.form.get('features')
        model_name = request.form.get('model')
        try:
            # Parse the input features (assuming numeric values separated by commas)
            features = [float(val.strip()) for val in features_str.split(',')]
            input_features = np.array(features).reshape(1, -1)
        except Exception as e:
            return jsonify({"error": "Invalid feature input. Please enter comma separated numbers."})
        
        if model_name not in best_models:
            return jsonify({"error": "Invalid model selection."})
        
        # Use the selected model to predict
        model = best_models[model_name]
        prediction = model.predict(input_features)[0]
        prediction_prob = None
        # If the model supports probability prediction, add the probability score
        if hasattr(model.named_steps['classifier'], "predict_proba"):
            prediction_prob = model.predict_proba(input_features).max()
        
        return jsonify({
            "model": model_name,
            "prediction": int(prediction),
            "probability": prediction_prob
        })
    else:
        # Render the prediction form and send available models for selection
        return render_template('predict.html', models=list(best_models.keys()))

# Function to run Flask app on port 8000 in a thread
def run_app():
    app.run(debug=True, use_reloader=False, port=8000)

# Start Flask app in a separate thread if running as a script
if __name__ == '__main__':
    threading.Thread(target=run_app).start()
