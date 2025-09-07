# Crop Monitoring and Prediction System using AI & Remote Sensing

## Project Overview
This project implements an advanced AI-driven platform for crop health monitoring, pest identification, and yield prediction by integrating multispectral/hyperspectral satellite data, IoT sensor inputs, and machine learning models. The system leverages state-of-the-art deep learning, classical machine learning, and explainability techniques to deliver real-time actionable agricultural insights.

---

## Key Features
- **Crop Health Monitoring:** Uses vegetation indices (NDVI, EVI, NDWI, MSAVI2) computed from hyperspectral and satellite imagery to assess crop vigor and water content.
- **Pest Identification:** Employs deep learning models with transfer learning (EfficientNet, ResNet, MobileNet) to detect and classify pest species from insect images.
- **Yield Prediction:** Combines remote sensing indices and environmental sensor data in regression models (Random Forest, XGBoost) and interpretable formulas for accurate crop yield forecasts.
- **Temporal Analysis:** Applies LSTM-based models to analyze and forecast temporal trends in crop growth and environmental variables.
- **Multimodal Fusion:** Integrates hyperspectral imagery with sensor data in a deep neural net to enhance classification accuracy.
- **User Interface:** Interactive Streamlit dashboard with visualization, data input, prediction capabilities, and chatbot assistance for farmers.

---

## Datasets Used
- **Salinas Hyperspectral Dataset:** High-resolution hyperspectral imagery with ground truth for crop classification.
- **Crop Yield & Disease Data:** Tabular data containing crop yields and associated environmental and management variables.
- **Farm Insects Image Dataset:** Diverse images of pest insects with labeled classes for pest detection.
- **Satellite Data:**
  - Sentinel-2 MSI (SAFE format): Multispectral satellite imagery used for computing vegetation indices.
  - Landsat 8 OLI: Satellite imagery for additional spectral information.

---

## Model Highlights & Achievements

- **Best Crop Classification:**  
  Random Forest trained on hyperspectral Salinas data achieved high accuracy, providing reliable pixel-level crop type classification.

- **Vegetation Indices:**  
  NDVI, EVI, NDWI, and MSAVI2 calculated from multispectral and hyperspectral input data for precise vegetation and moisture status mapping.

- **Yield Prediction:**  
  Used an interpretable linear regression formula combining NDVI, soil moisture, rainfall, temperature, nitrogen, and phosphorus levels. Supplemented with Random Forest and XGBoost regression models showing improved prediction performance.

- **Pest Identification Challenges:**  
  Despite efforts with current CNN architectures, pest classification accuracy remained suboptimal due to:
  - Limited labeled images for certain pest categories.
  - Visual similarity between pest species causing confusion.
  - Environmental variations, occlusion, and poor image quality.

- **Solutions & Improvements for Pest Model:**  
  - Planned upgrade to more powerful CNN architectures like **AlexNet** for better feature learning.  
  - Augmentation and significant expansion of pest image datasets to improve variability and generalization.  
  - Utilization of class balancing techniques and informed pre-processing pipelines.

---

## Technical Stack

- **Programming Languages:** Python 3.x  
- **ML Libraries:** TensorFlow, Keras, scikit-learn, XGBoost  
- **Data Processing:** NumPy, Pandas, Rasterio, OpenCV  
- **Visualization:** Matplotlib, Seaborn, Plotly, Streamlit  
- **Geospatial Tools:** Rasterio, EarthPy, GeoPandas  
- **Deployment:** Streamlit Web App with interactive dashboards and chatbot interface

---

## Future Work & Recommendations

- Enhance pest detection accuracy by applying **AlexNet** with larger, more diverse datasets.  
- Incorporate **explainable AI (XAI)** methods to improve model transparency and farmer trust.  
- Extend temporal modeling with more advanced LSTM or attention-based architectures.  
- Optimize deployment for edge devices in low-connectivity areas to enable wider real-world adoption.  
- Expand datasets to cover more crop types, pests, and environmental conditions.

---

## Credits & References

- Salinas and Sentinel-2 datasets from public remote sensing open sources.  
- Inspired by state-of-the-art AI research in agricultural monitoring and forecasting.  
- Developed with support from academic and open-source communities.

---
