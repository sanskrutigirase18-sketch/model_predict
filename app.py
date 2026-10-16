import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Attractive, modern responsive UI template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #cbd5e1;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
        }
        h1 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 8px;
            font-size: 24px;
        }
        p.subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 13px;
            color: #475569;
        }
        input, select {
            padding: 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.2s, box-shadow 0.2s;
            outline: none;
        }
        input:focus, select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .result-card h2 {
            margin: 0 0 5px 0;
            color: var(--primary-color);
            font-size: 18px;
        }
        .result-card p {
            margin: 0;
            font-size: 26px;
            font-weight: bold;
            color: #312e81;
        }
        .error {
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
            grid-column: span 2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Performance Predictor</h1>
        <p class="subtitle">Enter student activity metrics to calculate the predicted score.</p>
        
        <form method="POST">
            <div class="form-grid">
                {% if error %}
                <div class="error">{{ error }}</div>
                {% endif %}

                <div class="form-group">
                    <label for="study_hours">Study Hours (per week)</label>
                    <input type="number" step="0.1" id="study_hours" name="study_hours" required placeholder="e.g., 15">
                </div>
                <div class="form-group">
                    <label for="class_attendance">Class Attendance (%)</label>
                    <input type="number" step="0.1" id="class_attendance" name="class_attendance" required placeholder="e.g., 85">
                </div>
                <div class="form-group">
                    <label for="sleep_hours">Sleep Hours (per night)</label>
                    <input type="number" step="0.1" id="sleep_hours" name="sleep_hours" required placeholder="e.g., 7">
                </div>
                <div class="form-group">
                    <label for="sleep_quality">Sleep Quality (1-10)</label>
                    <input type="number" id="sleep_quality" name="sleep_quality" required placeholder="e.g., 8">
                </div>
                <div class="form-group">
                    <label for="facility_rating">Facility Rating (1-5)</label>
                    <input type="number" id="facility_rating" name="facility_rating" required placeholder="e.g., 4">
                </div>
                <div class="form-group">
                    <label for="exam_difficulty">Exam Difficulty (1-5)</label>
                    <input type="number" id="exam_difficulty" name="exam_difficulty" required placeholder="e.g., 3">
                </div>
                
                <button type="submit">Predict Score</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h2>Estimated Prediction Result</h2>
            <p>{{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            # Extract inputs from form fields
            study_hours = float(request.form["study_hours"])
            class_attendance = float(request.form["class_attendance"])
            sleep_hours = float(request.form["sleep_hours"])
            sleep_quality = float(request.form["sleep_quality"])
            facility_rating = float(request.form["facility_rating"])
            exam_difficulty = float(request.form["exam_difficulty"])
            
            # Format inputs into the array format expected by the model
            features = np.array([[study_hours, class_attendance, sleep_hours, sleep_quality, facility_rating, exam_difficulty]])
            
            if model is None:
                error = "Model file could not be loaded on the server."
            else:
                pred = model.predict(features)
                prediction = round(float(pred[0]), 2)
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
