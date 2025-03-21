import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Trained Model
model = joblib.load("disease_prediction_model.joblib")

# Load Symptom Names from Training Data
# Note: Ensure this matches the symptom order used during training
# You can load the CSV to get exact symptom names
df = pd.read_csv("BinaryFeatures_DiseaseAndSymptoms.csv")
symptom_columns = list(df.columns[1:])  # Exclude 'Disease' column

# Streamlit App Title
st.title("🩺 Disease Prediction Based on Symptoms")
st.write("Select the symptoms you're experiencing to predict the likely disease.")

# Multi-Select Symptoms
selected_symptoms = st.multiselect(
    "Choose your symptoms:",
    symptom_columns
)

# Create Input Vector
input_vector = np.zeros(len(symptom_columns))

# Set selected symptoms to 1
for symptom in selected_symptoms:
    index = symptom_columns.index(symptom)
    input_vector[index] = 1

# Predict Button
if st.button("Predict Disease"):
    prediction = model.predict([input_vector])[0]
    st.success(f"🧾 Predicted Disease: **{prediction}**")

    # Optional: Show prediction probability
    probabilities = model.predict_proba([input_vector])[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    st.subheader("🔍 Top 3 Prediction Probabilities:")
    for idx in top_3_indices:
        disease = model.classes_[idx]
        prob = probabilities[idx] * 100
        st.write(f"- {disease}: {prob:.2f}%")

# Footer
st.markdown("---")
st.markdown("Developed by Nishu Pandey")
