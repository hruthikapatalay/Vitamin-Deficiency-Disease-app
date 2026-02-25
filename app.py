import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# LOAD MODEL & PREPROCESSOR
# -------------------------------
model = joblib.load("rf_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Vitamin Deficiency Predictor", layout="wide")

st.title("🩺 Vitamin Deficiency Disease Prediction App")
st.markdown("Enter patient clinical and lifestyle details below.")

# -------------------------------
# SECTION 1 — BASIC INFO
# -------------------------------
st.subheader("👤 Basic Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 25)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col3:
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

# -------------------------------
# SECTION 2 — LIFESTYLE
# -------------------------------
st.subheader("🏃 Lifestyle Factors")
col1, col2, col3 = st.columns(3)

with col1:
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])

with col2:
    exercise_level = st.selectbox("Exercise Level", ["Active", "Moderate", "Light"])
    diet_type = st.selectbox("Diet Type", ["Vegetarian", "Omnivore", "Pescatarian"])

with col3:
    sun_exposure = st.selectbox("Sun Exposure", ["High", "Moderate", "Low"])
    income_level = st.selectbox("Income Level", ["High", "Middle", "Low"])
    latitude_region = st.selectbox("Latitude Region", ["High", "Middle", "Low"])

# -------------------------------
# SECTION 3 — VITAMIN LEVELS
# -------------------------------
st.subheader("🧪 Vitamin & Lab Values")
col1, col2, col3 = st.columns(3)

with col1:
    vitamin_a_percent_rda = st.number_input("Vitamin A % RDA", 0.0, 200.0, 80.0)
    vitamin_c_percent_rda = st.number_input("Vitamin C % RDA", 0.0, 200.0, 80.0)
    vitamin_d_percent_rda = st.number_input("Vitamin D % RDA", 0.0, 200.0, 80.0)

with col2:
    vitamin_e_percent_rda = st.number_input("Vitamin E % RDA", 0.0, 200.0, 80.0)
    vitamin_b12_percent_rda = st.number_input("Vitamin B12 % RDA", 0.0, 200.0, 80.0)
    folate_percent_rda = st.number_input("Folate % RDA", 0.0, 200.0, 80.0)

with col3:
    calcium_percent_rda = st.number_input("Calcium % RDA", 0.0, 200.0, 80.0)
    iron_percent_rda = st.number_input("Iron % RDA", 0.0, 200.0, 80.0)
    hemoglobin_g_dl = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.0)

serum_vitamin_d_ng_ml = st.number_input("Serum Vitamin D (ng/mL)", 0.0, 100.0, 30.0)
serum_vitamin_b12_pg_ml = st.number_input("Serum Vitamin B12 (pg/mL)", 0.0, 1000.0, 400.0)
serum_folate_ng_ml = st.number_input("Serum Folate (ng/mL)", 0.0, 50.0, 10.0)

symptoms_count = st.number_input("Symptoms Count", 0, 20, 2)
symptoms_list = st.selectbox("Primary Symptom", 
    ["None", "Fatigue", "Bone Pain", "Night Blindness", "Bleeding Gums", "Memory Problems"]
)

# -------------------------------
# SECTION 4 — SYMPTOMS (Binary)
# -------------------------------
st.subheader("⚠ Symptoms")
col1, col2, col3 = st.columns(3)

with col1:
    has_night_blindness = st.selectbox("Night Blindness", [0, 1])
    has_fatigue = st.selectbox("Fatigue", [0, 1])
    has_bleeding_gums = st.selectbox("Bleeding Gums", [0, 1])

with col2:
    has_bone_pain = st.selectbox("Bone Pain", [0, 1])
    has_muscle_weakness = st.selectbox("Muscle Weakness", [0, 1])
    has_numbness_tingling = st.selectbox("Numbness/Tingling", [0, 1])

with col3:
    has_memory_problems = st.selectbox("Memory Problems", [0, 1])
    has_pale_skin = st.selectbox("Pale Skin", [0, 1])
    has_multiple_deficiencies = st.selectbox("Multiple Deficiencies", [0, 1])

# -------------------------------
# CREATE DATAFRAME
# -------------------------------
input_dict = {
    'age': age,
    'gender': gender,
    'bmi': bmi,
    'smoking_status': smoking_status,
    'alcohol_consumption': alcohol_consumption,
    'exercise_level': exercise_level,
    'diet_type': diet_type,
    'sun_exposure': sun_exposure,
    'income_level': income_level,
    'latitude_region': latitude_region,
    'vitamin_a_percent_rda': vitamin_a_percent_rda,
    'vitamin_c_percent_rda': vitamin_c_percent_rda,
    'vitamin_d_percent_rda': vitamin_d_percent_rda,
    'vitamin_e_percent_rda': vitamin_e_percent_rda,
    'vitamin_b12_percent_rda': vitamin_b12_percent_rda,
    'folate_percent_rda': folate_percent_rda,
    'calcium_percent_rda': calcium_percent_rda,
    'iron_percent_rda': iron_percent_rda,
    'hemoglobin_g_dl': hemoglobin_g_dl,
    'serum_vitamin_d_ng_ml': serum_vitamin_d_ng_ml,
    'serum_vitamin_b12_pg_ml': serum_vitamin_b12_pg_ml,
    'serum_folate_ng_ml': serum_folate_ng_ml,
    'symptoms_count': symptoms_count,
    'symptoms_list': symptoms_list,
    'has_night_blindness': has_night_blindness,
    'has_fatigue': has_fatigue,
    'has_bleeding_gums': has_bleeding_gums,
    'has_bone_pain': has_bone_pain,
    'has_muscle_weakness': has_muscle_weakness,
    'has_numbness_tingling': has_numbness_tingling,
    'has_memory_problems': has_memory_problems,
    'has_pale_skin': has_pale_skin,
    'has_multiple_deficiencies': has_multiple_deficiencies
}

input_data = pd.DataFrame([input_dict])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Disease"):
    try:
        # Ensure column order matches preprocessor
        input_data = input_data[preprocessor.feature_names_in_]

        # Convert numeric columns to float to avoid dtype errors
        numeric_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
        input_data[numeric_cols] = input_data[numeric_cols].astype(float)

        # Transform and predict
        processed = preprocessor.transform(input_data)
        prediction = model.predict(processed)
        probability = model.predict_proba(processed)

        st.success(f"🧾 Predicted Disease: {prediction[0]}")
        st.write("Prediction Confidence:")
        st.write(dict(zip(model.classes_, probability[0])))
    except Exception as e:
        st.error(f"Error in preprocessing/prediction: {e}")