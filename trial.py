import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
# Load trained model
MODEL_PATH = "disease_prediction_model.joblib"
model = joblib.load(MODEL_PATH)

# Load symptom names from CSV
df = pd.read_csv("BinaryFeatures_DiseaseAndSymptoms.csv")
symptom_columns = list(df.columns[1:])  # Exclude 'Disease' column

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Disease Prediction API is running!"

# ✅ API to get symptoms list
@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptom_columns})

# ✅ API to predict disease
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        selected_symptoms = data.get("symptoms", [])
        
        if not selected_symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Create input vector
        input_vector = np.zeros(len(symptom_columns))
        for symptom in selected_symptoms:
            if symptom in symptom_columns:
                index = symptom_columns.index(symptom)
                input_vector[index] = 1

        # Make prediction
        predicted_disease = model.predict([input_vector])[0]

        # Get top 3 probabilities
        probabilities = model.predict_proba([input_vector])[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {"disease": model.classes_[idx], "probability": f"{probabilities[idx] * 100:.2f}%"}
            for idx in top_3_indices
        ]

        return jsonify({
            "predicted_disease": predicted_disease,
            "top_3_predictions": top_3_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == "__main__":
    app.run(debug=True)
