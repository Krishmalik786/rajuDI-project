import streamlit as st
import requests
import json
import pandas as pd

# Page config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Breast Cancer Prediction System")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 About")
st.sidebar.info(
    """
    This ML application predicts whether a breast tumor is:
    - **Malignant** (Cancerous)
    - **Benign** (Non-cancerous)
    
    Based on tumor cell characteristics.
    """
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ API Status")

# Check API health
API_URL = "http://localhost:8000"

try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Not Running")
    st.error("⚠️ Please start the API server first: `uvicorn deployment.app:app --reload`")
    st.stop()

# Main content
st.header("🔬 Enter Tumor Features")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mean Values")
    mean_radius = st.number_input("Mean Radius", value=17.99, format="%.2f")
    mean_texture = st.number_input("Mean Texture", value=10.38, format="%.2f")
    mean_perimeter = st.number_input("Mean Perimeter", value=122.8, format="%.2f")
    mean_area = st.number_input("Mean Area", value=1001.0, format="%.2f")
    mean_smoothness = st.number_input("Mean Smoothness", value=0.1184, format="%.4f")
    mean_compactness = st.number_input("Mean Compactness", value=0.2776, format="%.4f")
    mean_concavity = st.number_input("Mean Concavity", value=0.3001, format="%.4f")
    mean_concave_points = st.number_input("Mean Concave Points", value=0.1471, format="%.4f")
    mean_symmetry = st.number_input("Mean Symmetry", value=0.2419, format="%.4f")
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.07871, format="%.5f")

with col2:
    st.subheader("Error Values")
    radius_error = st.number_input("Radius Error", value=1.095, format="%.3f")
    texture_error = st.number_input("Texture Error", value=0.9053, format="%.4f")
    perimeter_error = st.number_input("Perimeter Error", value=8.589, format="%.3f")
    area_error = st.number_input("Area Error", value=153.4, format="%.2f")
    smoothness_error = st.number_input("Smoothness Error", value=0.006399, format="%.6f")
    compactness_error = st.number_input("Compactness Error", value=0.04904, format="%.5f")
    concavity_error = st.number_input("Concavity Error", value=0.05373, format="%.5f")
    concave_points_error = st.number_input("Concave Points Error", value=0.01587, format="%.5f")
    symmetry_error = st.number_input("Symmetry Error", value=0.03003, format="%.5f")
    fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.006193, format="%.6f")

# Third column for worst values
st.subheader("Worst Values")
col3, col4 = st.columns(2)

with col3:
    worst_radius = st.number_input("Worst Radius", value=25.38, format="%.2f")
    worst_texture = st.number_input("Worst Texture", value=17.33, format="%.2f")
    worst_perimeter = st.number_input("Worst Perimeter", value=184.6, format="%.2f")
    worst_area = st.number_input("Worst Area", value=2019.0, format="%.2f")
    worst_smoothness = st.number_input("Worst Smoothness", value=0.1622, format="%.4f")

with col4:
    worst_compactness = st.number_input("Worst Compactness", value=0.6656, format="%.4f")
    worst_concavity = st.number_input("Worst Concavity", value=0.7119, format="%.4f")
    worst_concave_points = st.number_input("Worst Concave Points", value=0.2654, format="%.4f")
    worst_symmetry = st.number_input("Worst Symmetry", value=0.4601, format="%.4f")
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.1189, format="%.4f")

st.markdown("---")

# Predict button
if st.button("🔮 Predict", type="primary", use_container_width=True):
    # Prepare data
    data = {
        "mean_radius": mean_radius,
        "mean_texture": mean_texture,
        "mean_perimeter": mean_perimeter,
        "mean_area": mean_area,
        "mean_smoothness": mean_smoothness,
        "mean_compactness": mean_compactness,
        "mean_concavity": mean_concavity,
        "mean_concave_points": mean_concave_points,
        "mean_symmetry": mean_symmetry,
        "mean_fractal_dimension": mean_fractal_dimension,
        "radius_error": radius_error,
        "texture_error": texture_error,
        "perimeter_error": perimeter_error,
        "area_error": area_error,
        "smoothness_error": smoothness_error,
        "compactness_error": compactness_error,
        "concavity_error": concavity_error,
        "concave_points_error": concave_points_error,
        "symmetry_error": symmetry_error,
        "fractal_dimension_error": fractal_dimension_error,
        "worst_radius": worst_radius,
        "worst_texture": worst_texture,
        "worst_perimeter": worst_perimeter,
        "worst_area": worst_area,
        "worst_smoothness": worst_smoothness,
        "worst_compactness": worst_compactness,
        "worst_concavity": worst_concavity,
        "worst_concave_points": worst_concave_points,
        "worst_symmetry": worst_symmetry,
        "worst_fractal_dimension": worst_fractal_dimension
    }
    
    # Make prediction
    with st.spinner("Making prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                st.markdown("---")
                st.header("📊 Prediction Result")
                
                # Create columns for result display
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(
                        label="Prediction",
                        value=result['prediction_label']
                    )
                
                with res_col2:
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']*100:.2f}%"
                    )
                
                with res_col3:
                    if result['prediction_label'] == 'Benign':
                        st.success("✅ Non-Cancerous")
                    else:
                        st.error("⚠️ Cancerous")
                
                # Probability breakdown
                st.markdown("---")
                st.subheader("📈 Probability Breakdown")
                
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.info(f"**Malignant:** {result['probability_malignant']*100:.2f}%")
                
                with prob_col2:
                    st.info(f"**Benign:** {result['probability_benign']*100:.2f}%")
                
                # Progress bars
                st.progress(result['probability_malignant'], text="Malignant Probability")
                st.progress(result['probability_benign'], text="Benign Probability")
                
                # Disclaimer
                st.markdown("---")
                st.warning("⚕️ **Medical Disclaimer:** This is a machine learning prediction for educational purposes. Always consult healthcare professionals for medical diagnosis.")
                
            else:
                st.error(f"❌ Prediction failed: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using FastAPI, Streamlit & MLflow</p>
        <p>MLOps Project | 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)
