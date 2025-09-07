import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
import requests
import time

# Set up the Streamlit app
st.set_page_config(page_title="Crop Health Monitoring", layout="wide")

# --- Custom CSS for Styling and Animations ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fade-in animation for all content */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease forwards;
    }
    
    /* Main background with a vibrant color and subtle glow */
    .main {
        background: #0D1117;
        color: #f0f2f6;
        position: relative;
    }
    .main::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 800px;
        height: 800px;
        background: radial-gradient(circle, rgba(15, 185, 178, 0.1), transparent 70%);
        filter: blur(80px);
        transform: translate(-50%, -50%);
        z-index: 0;
    }
    
    /* Page background with very faint purple gradient */
    .stApp {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.97) 0%, rgba(88, 28, 135, 0.05) 50%, rgba(15, 23, 42, 0.97) 100%);
    }
    
    /* Pink to purple gradient for all headings */
    .pink-purple-gradient {
        background: linear-gradient(135deg, #FF6EC7 0%, #7873FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Home page title gradient */
    .title-gradient {
        background: linear-gradient(135deg, #FF6EC7 0%, #7873FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 5rem;
        font-weight: 700;
        letter-spacing: -2px;
    }

    .home-content {
        text-align: left;
    }
    
    /* White text for the subtitle */
    .white-text {
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Centered sidebar title */
    .sidebar-title {
        text-align: center;
        margin-bottom: 30px;
        font-size: 4rem;
        background: linear-gradient(135deg, #FF6EC7 0%, #7873FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    
    /* Cards for metrics and sections */
    .stMetric, .stAlert, .stSubheader, .stHeader {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
        border: 1px solid #333;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.15);
    }
    /* Metric values with gradients */
    [data-testid="stMetricValue"] {
        background: -webkit-linear-gradient(left, #1c92d2, #f2fcfe);
        background: linear-gradient(to right, #1c92d2, #f2fcfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 2.5rem;
    }
    [data-testid="stMetricDelta"] {
        color: #4CAF50;
        font-weight: 600;
    }
    /* Chatbot styling */
    .stChatInput, .stChatMsg, .st-b {
        border-radius: 10px;
        border: 1px solid #444;
        background-color: #222;
        color: #f0f2f6;
    }
    
    /* Animation delays for different elements */
    .animate-1 { animation-delay: 0.1s; }
    .animate-2 { animation-delay: 0.2s; }
    .animate-3 { animation-delay: 0.3s; }
    .animate-4 { animation-delay: 0.4s; }
    .animate-5 { animation-delay: 0.5s; }
</style>
""", unsafe_allow_html=True)

# --- Gemini API Integration ---
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
API_KEY = "AIzaSyAAVCx8NNpcAZdU0qBYjPiCowfBHZi1xac"

def get_gemini_response(prompt, image_data=None):
    """Sends a request to the Gemini API and returns the response."""
    headers = {
        'Content-Type': 'application/json'
    }

    contents = []
    # Add text prompt
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    # Add image if available
    if image_data:
        contents[0]["parts"].append({
            "inlineData": {
                "mimeType": "image/jpeg",  # Assuming jpeg for most use cases
                "data": image_data
            }
        })
    
    payload = {
        "contents": contents
    }

    try:
        # Use the API_URL with the provided key
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as err:
        return f"HTTP Error: {err}"
    except Exception as err:
        return f"An error occurred: {err}"


# Sidebar for navigation
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-size: 2.5rem; background: linear-gradient(135deg, #FF6EC7 0%, #7873FF 100%); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; 
        font-weight: bold;'>agriIntel</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Select a page:", 
                               ["Home", "Dashboard", "Vegetation Indices", "Pest Detection", "Yield Prediction", "Temporal Analysis", "Chatbot"])

# Initialize session state for page tracking
if 'current_page' not in st.session_state:
    st.session_state.current_page = options

# Check if page has changed
page_changed = st.session_state.current_page != options
st.session_state.current_page = options

if options == "Home":
    st.markdown('<div class="home-content fade-in"><div class="title-gradient">agriIntel</div></div>', unsafe_allow_html=True)
    st.markdown("## <span class='white-text fade-in animate-1'>AI-driven intelligence for modern agriculture.</span>", unsafe_allow_html=True)
    st.write("AgriIntel combines advanced analytics with agronomy to help you make sharper, greener decisions.")

elif options == "Dashboard":
    st.markdown("<h1 class='pink-purple-gradient fade-in'>Farm Dashboard</h1>", unsafe_allow_html=True)
    
    # Create two columns with adjusted ratio
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-1'>Field Overview</h2>", unsafe_allow_html=True)
        # Display map or field image
        st.image("WhatsApp Image 2025-09-07 at 07.12.44.jpeg", use_container_width=True)
        
    with col2:
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-1'>Key Metrics</h2>", unsafe_allow_html=True)
        
        # Create a container for metrics with reduced spacing
        metrics_container = st.container()
        with metrics_container:
            # Use columns for metrics to save vertical space
            mcol1, mcol2 = st.columns(2)
            
            with mcol1:
                st.markdown('<div class="fade-in animate-2">', unsafe_allow_html=True)
                st.metric("Average NDVI", "0.68", "0.12")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with mcol2:
                st.markdown('<div class="fade-in animate-3">', unsafe_allow_html=True)
                st.metric("Pest Risk", "Low", "-2%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Single column for the last metric to make it centered
            st.markdown('<div class="fade-in animate-4">', unsafe_allow_html=True)
            st.metric("Yield Forecast", "5.2 tons/acre", "0.3 tons")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some custom CSS to reduce the metrics size
        st.markdown("""
        <style>
            .stMetric {
                padding: 30px !important;
                margin-bottom: 16px !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 3rem !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Recent alerts
    st.markdown("<h2 class='pink-purple-gradient fade-in animate-5'>Recent Alerts</h2>", unsafe_allow_html=True)
    
    # Use columns for alerts to save space
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
        st.warning("Northern section shows signs of water stress")
        st.markdown('</div>', unsafe_allow_html=True)
    with alert_col2:
        st.markdown('<div class="fade-in animate-2">', unsafe_allow_html=True)
        st.info("Soil moisture optimal in southern fields")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # NDVI trend
    st.markdown("<h2 class='pink-purple-gradient fade-in animate-3'>NDVI Trend (Last 30 Days)</h2>", unsafe_allow_html=True)
    trend_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=30),
        'NDVI': np.random.uniform(0.6, 0.8, 30)
    })
    fig = px.line(trend_data, x='Date', y='NDVI', title='NDVI Trend Over Time')
    st.plotly_chart(fig, use_container_width=True)

elif options == "Vegetation Indices":
    st.markdown("<h1 class='pink-purple-gradient fade-in'>Vegetation Indices Analysis</h1>", unsafe_allow_html=True)
    
    # Upload multispectral image
    st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload multispectral image", type=["tif", "jpg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process image and calculate indices
        st.markdown('<div class="fade-in animate-2">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display indices (placeholder)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="fade-in animate-3">', unsafe_allow_html=True)
            st.markdown("<h3 class='pink-purple-gradient'>NDVI</h3>", unsafe_allow_html=True)
            st.image("https://via.placeholder.com/200x150?text=NDVI+Map", caption="NDVI Map", use_container_width=True)
            st.metric("Average NDVI", "0.72", "0.05")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="fade-in animate-4">', unsafe_allow_html=True)
            st.markdown("<h3 class='pink-purple-gradient'>EVI</h3>", unsafe_allow_html=True)
            st.image("https://via.placeholder.com/200x150?text=EVI+Map", caption="EVI Map", use_container_width=True)
            st.metric("Average EVI", "0.45", "0.03")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="fade-in animate-5">', unsafe_allow_html=True)
            st.markdown("<h3 class='pink-purple-gradient'>NDWI</h3>", unsafe_allow_html=True)
            st.image("https://via.placeholder.com/200x150?text=NDWI+Map", caption="NDWI Map", use_container_width=True)
            st.metric("Average NDWI", "0.28", "-0.02")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
            st.markdown("<h3 class='pink-purple-gradient'>MSAVI2</h3>", unsafe_allow_html=True)
            st.image("https://via.placeholder.com/200x150?text=MSAVI2+Map", caption="MSAVI2 Map", use_container_width=True)
            st.metric("Average MSAVI2", "0.62", "0.04")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show health status
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-2'>Field Health Status</h2>", unsafe_allow_html=True)
        health_data = pd.DataFrame({
            'Section': ['North', 'South', 'East', 'West'],
            'NDVI': [0.68, 0.72, 0.65, 0.71],
            'Health': ['Moderate', 'Good', 'Moderate', 'Good']
        })
        fig = px.bar(health_data, x='Section', y='NDVI', color='Health', 
                     title='NDVI by Field Section')
        st.plotly_chart(fig, use_container_width=True)

elif options == "Pest Detection":
    st.markdown("<h1 class='pink-purple-gradient fade-in'>Pest Detection</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload insect image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown('<div class="fade-in animate-2">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Insect Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Placeholder for pest detection results with confidence heatmap
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-3'>Detection Results</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="fade-in animate-4">', unsafe_allow_html=True)
            st.write("*Identified Pest:* Aphids")
            st.write("*Confidence:* 87%")
            st.write("*Risk Level:* High")
            st.write("- Apply neem oil solution")
            st.write("- Introduce ladybugs as natural predators")
            st.write("- Monitor affected area daily")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="fade-in animate-5">', unsafe_allow_html=True)
            # Confidence heatmap (placeholder)
            st.write("*Detection Confidence:*")
            pest_confidences = {
                'Aphids': 0.87,
                'Whiteflies': 0.08,
                'Spider Mites': 0.03,
                'Other': 0.02
            }
            fig = px.bar(x=list(pest_confidences.keys()), y=list(pest_confidences.values()),
                         labels={'x': 'Pest Type', 'y': 'Confidence'},
                         title='Pest Detection Confidence')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show historical pest occurrences
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-1'>Historical Pest Occurrences</h2>", unsafe_allow_html=True)
        history_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'Aphids': np.random.randint(0, 10, 12),
            'Whiteflies': np.random.randint(0, 5, 12),
            'Spider Mites': np.random.randint(0, 3, 12)
        })
        fig = px.line(history_data, x='Date', y=['Aphids', 'Whiteflies', 'Spider Mites'],
                      title='Pest Occurrences Over Time')
        st.plotly_chart(fig, use_container_width=True)

elif options == "Yield Prediction":
    st.markdown("<h1 class='pink-purple-gradient fade-in'>Yield Prediction</h1>", unsafe_allow_html=True)
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
        ndvi = st.slider("Average NDVI", 0.0, 1.0, 0.7)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="fade-in animate-2">', unsafe_allow_html=True)
        soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 65.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="fade-in animate-3">', unsafe_allow_html=True)
        rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 120.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="fade-in animate-4">', unsafe_allow_html=True)
        temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="fade-in animate-5">', unsafe_allow_html=True)
        nitrogen = st.slider("Nitrogen Level (kg/ha)", 0.0, 200.0, 100.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
        phosphorus = st.slider("Phosphorus Level (kg/ha)", 0.0, 100.0, 50.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict yield
    st.markdown('<div class="fade-in animate-2">', unsafe_allow_html=True)
    if st.button("Predict Yield"):
        # Enhanced prediction formula
        predicted_yield = (2.5 + (ndvi * 3.2) + (soil_moisture * 0.02) + 
                          (rainfall * 0.005) - (temperature * 0.03) + 
                          (nitrogen * 0.01) + (phosphorus * 0.015))
        
        st.success(f"Predicted Yield: {predicted_yield:.2f} tons/acre")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show historical comparison
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-3'>Historical Comparison</h2>", unsafe_allow_html=True)
        historical_data = pd.DataFrame({
            'Year': [2018, 2019, 2020, 2021, 2022, 2023],
            'Yield': [4.2, 4.5, 4.8, 5.1, 4.9, predicted_yield]
        })
        fig = px.line(historical_data, x='Year', y='Yield', 
                     title='Historical Yield Comparison', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature importance
        st.markdown("<h2 class='pink-purple-gradient fade-in animate-4'>Feature Importance</h2>", unsafe_allow_html=True)
        importance_data = pd.DataFrame({
            'Feature': ['NDVI', 'Soil Moisture', 'Rainfall', 'Temperature', 'Nitrogen', 'Phosphorus'],
            'Importance': [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
        })
        fig = px.bar(importance_data, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance for Yield Prediction')
        st.plotly_chart(fig, use_container_width=True)

elif options == "Temporal Analysis":
    st.markdown("<h1 class='pink-purple-gradient fade-in'>Temporal Analysis</h1>", unsafe_allow_html=True)
    
    # Generate sample temporal data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    temporal_data = pd.DataFrame({
        'Date': dates,
        'NDVI': 0.6 + 0.2 * np.sin(np.arange(len(dates)) / 30) + np.random.normal(0, 0.05, len(dates)),
        'Temperature': 20 + 10 * np.sin(np.arange(len(dates)) / 30) + np.random.normal(0, 2, len(dates)),
        'Rainfall': np.random.exponential(2, len(dates)),
        'Soil Moisture': 50 + 20 * np.sin(np.arange(len(dates)) / 30) + np.random.normal(0, 5, len(dates))
    })
    
    # NDVI trend
    st.markdown("<h2 class='pink-purple-gradient fade-in animate-1'>NDVI Trend Over Time</h2>", unsafe_allow_html=True)
    fig = px.line(temporal_data, x='Date', y='NDVI', title='NDVI Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Multiple metrics
    st.markdown("<h2 class='pink-purple-gradient fade-in animate-2'>Multiple Metrics Over Time</h2>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=2, subplot_titles=('NDVI', 'Temperature', 'Rainfall', 'Soil Moisture'))
    
    fig.add_trace(go.Scatter(x=temporal_data['Date'], y=temporal_data['NDVI'], name='NDVI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=temporal_data['Date'], y=temporal_data['Temperature'], name='Temperature'), row=1, col=2)
    fig.add_trace(go.Bar(x=temporal_data['Date'], y=temporal_data['Rainfall'], name='Rainfall'), row=2, col=1)
    fig.add_trace(go.Scatter(x=temporal_data['Date'], y=temporal_data['Soil Moisture'], name='Soil Moisture'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("<h2 class='pink-purple-gradient fade-in animate-3'>Correlation Analysis</h2>", unsafe_allow_html=True)
    corr_matrix = temporal_data.drop('Date', axis=1).corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                   title='Correlation Between Metrics')
    st.plotly_chart(fig, use_container_width=True)

elif options == "Chatbot":
    st.markdown("<h1 class='pink-purple-gradient fade-in'>Agriculture Assistant Chatbot</h1>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "text" in message:
                st.markdown(message["text"])
            if "image" in message:
                st.image(message["image"], use_container_width=True)
    
    # React to user input
    st.markdown('<div class="fade-in animate-1">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask about crop health, pests, or yield")

    if uploaded_image or prompt:
        
        # Add user message to chat history
        with st.chat_message("user"):
            if prompt:
                st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "text": prompt})
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)
                st.session_state.messages.append({"role": "user", "image": uploaded_image})
        
        with st.spinner("Thinking..."):
            image_data = None
            if uploaded_image:
                image_data = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
                response = get_gemini_response(f"Analyze the following image for agricultural purposes and provide a report on pests, crop health, etc.: {prompt or ''}", image_data)
            else:
                response = get_gemini_response(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "text": response})
        
# Footer
st.sidebar.markdown("---")
st.sidebar.info("AI-Powered Agriculure Assitant | ©️ 2025")
