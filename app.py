import streamlit as st
import joblib
import pandas as pd
import os


st.set_page_config(
    page_title="Medical Cost Predictor",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def load_pipeline():
    pipeline_path = "medical_cost_pipeline_v1.joblib"
    if not os.path.exists(pipeline_path):
        st.error(f"Pipeline file '{pipeline_path}' not found. Please train the model first.")
        return None
    return joblib.load(pipeline_path)

pipeline = load_pipeline()


st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("""
**Estimate a new client's annual medical cost in under 3 minutes.**  
This tool uses a machine learning model trained on historical insurance data 
to predict costs based on client demographics, health profile, and plan selection.
""")

st.divider()


if pipeline is not None:
    with st.form("client_form"):
        st.subheader("📝 Client Information")
        
        
        st.markdown("#### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col3:
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            urban_rural = st.selectbox("Area Type", ["Urban", "Suburban", "Rural"])
        with col2:
            income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        with col3:
            education = st.selectbox("Education", [
                "No HS", "HS", "Some College", "Bachelor", "Master", "Doctorate"
            ])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            marital_status = st.selectbox("Marital Status", [
                "Single", "Married", "Divorced", "Widowed"
            ])
        with col2:
            employment_status = st.selectbox("Employment Status", [
                "Employed", "Self-employed", "Unemployed", "Retired"
            ])
        with col3:
            household_size = st.number_input("Household Size", min_value=1, value=2, step=1)
        
        dependents = st.number_input("Number of Dependents", min_value=0, value=0, step=1)
        
        st.divider()
        
     
        st.markdown("#### 🏃 Lifestyle")
        col1, col2 = st.columns(2)
        with col1:
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        with col2:
            smoker = st.selectbox("Smoker", ["No", "Yes", "Former"])
        
        alcohol_freq = st.selectbox("Alcohol Frequency", [
            "Never", "Occasionally", "Regularly", "Daily"
        ])
        
        st.divider()
        
      
        st.markdown("#### 🩺 Medical History")
        st.markdown("*Has the client been diagnosed with any of the following? (Yes = 1, No = 0)*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col2:
            copd = st.selectbox("COPD", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            cardiovascular_disease = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            cancer_history = st.selectbox("Cancer History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col3:
            kidney_disease = st.selectbox("Kidney Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            liver_disease = st.selectbox("Liver Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            arthritis = st.selectbox("Arthritis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        mental_health = st.selectbox("Mental Health Condition", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        hospitalizations_last_3yrs = st.selectbox(
            "Hospitalized in last 3 years?", 
            [0, 1, 2, 3], 
            format_func=lambda x: "No" if x == 0 else f"{x} time(s)"
        )
        
        st.divider()
        
        
        st.markdown("#### 📋 Insurance Plan Selection")
        col1, col2, col3 = st.columns(3)
        with col1:
            plan_type = st.selectbox("Plan Type", ["Bronze", "Silver", "Gold", "Platinum"])
        with col2:
            network_tier = st.selectbox("Network Tier", ["Basic", "Standard", "Premium"])
        with col3:
            policy_term_years = st.number_input("Policy Term (Years)", min_value=1, value=1, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            deductible = st.number_input("Deductible ($)", min_value=0, value=1500, step=100)
        with col2:
            copay = st.number_input("Copay ($)", min_value=0, value=30, step=5)
        
        st.divider()
        
      
        submitted = st.form_submit_button("🔮 Predict Annual Medical Cost", use_container_width=True)
    
  
    if submitted:
        
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "region": region,
            "urban_rural": urban_rural,
            "income": income,
            "education": education,
            "employment_status": employment_status,
            "marital_status": marital_status,
            "household_size": household_size,
            "dependents": dependents,
            "bmi": bmi,
            "smoker": smoker,
            "alcohol_freq": alcohol_freq,
            "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
            "plan_type": plan_type,
            "network_tier": network_tier,
            "deductible": deductible,
            "copay": copay,
            "policy_term_years": policy_term_years,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "asthma": asthma,
            "copd": copd,
            "cardiovascular_disease": cardiovascular_disease,
            "cancer_history": cancer_history,
            "kidney_disease": kidney_disease,
            "liver_disease": liver_disease,
            "arthritis": arthritis,
            "mental_health": mental_health,
        }])
        
      
        prediction = pipeline.predict(input_data)[0]
        
      
        st.success("## ✅ Prediction Complete")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Estimated Annual Medical Cost",
                value=f"${prediction:,.2f}",
                delta=f"${prediction/12:,.2f}/month"
            )
        
        st.info("""
        **Disclaimer:** This is an initial estimate based on self-reported information. 
        Final premium may be adjusted after medical underwriting and verification of 
        health records.
        """)