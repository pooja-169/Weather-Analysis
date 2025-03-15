# **Weather Data Analysis and Forecasting Project**

## **Project Overview**
This project analyzes global weather data, detects anomalies, performs time series forecasting, and explores spatial trends in climate variations. The dataset includes temperature, humidity, wind speed, pressure, and air quality indicators, which are used for data visualization, correlation analysis, and predictive modeling.

## **Dataset**
- **File Name:** `GlobalWeatherRepository.csv`
- **Source:** 'https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code'
- **Features:**
  - **Weather Data:** Temperature (Celsius & Fahrenheit), humidity, wind speed, pressure, precipitation.
  - **Air Quality Metrics:** PM2.5, PM10, Carbon Monoxide, Ozone, Nitrogen Dioxide, Sulfur Dioxide.
  - **Geospatial Information:** Latitude, Longitude.
  - **Temporal Information:** Timestamp of recorded observations.

## **Key Features and Analysis**
### **1. Exploratory Data Analysis (EDA)**
- Summary statistics to understand data distribution.
- Heatmaps to visualize feature correlations.
- Boxplots to detect and handle outliers.

### **2. Outlier Handling**
- Used **Interquartile Range (IQR)** method to detect extreme values.
- Outliers were replaced with median values to prevent bias while preserving meaningful data.

### **3. Data Normalization**
- Applied **MinMaxScaler** to scale numerical features between 0 and 1, ensuring consistency for machine learning models.

### **4. Time Series Forecasting**
- **ARIMA Model:** Used for forecasting temperature trends, though it struggled with seasonal variations.
- **Holt-Winters Model:** Performed better in detecting trends but slightly exaggerated the downward trend.

### **5. Anomaly Detection**
- **Isolation Forest & Local Outlier Factor (LOF)** used to detect extreme weather patterns such as sudden temperature spikes or pollution surges.

### **6. Machine Learning Models for Forecasting**
- **Random Forest, Linear Regression, Ridge Regression** used for predicting temperature with **0.99 R² scores**, making deep learning unnecessary.

### **7. Climate Trend Analysis**
- Long-term climate trend visualization showing temperature changes over the years.

### **8. Spatial Analysis**
- Scatter plots and **heatmaps** to visualize temperature, humidity, and air quality variations across different locations.

## **Installation & Requirements**
### **1. Install Dependencies**
To run this project, install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### **2. Running the Project**
- Load the dataset in a Jupyter Notebook or Python script.
- Execute each section for **EDA, outlier handling, forecasting, and anomaly detection**.
- Visualizations will be generated automatically for analysis.

## **Results & Insights**
- **Outlier handling improved data stability**, making models more reliable.
- **Holt-Winters performed better than ARIMA** in forecasting seasonal trends.
- **Random Forest achieved high accuracy** in temperature prediction.
- **Spatial analysis revealed temperature hotspots** and air quality variations.

## **Future Improvements**
- Integrate **deep learning models** (LSTMs) for improved forecasting.
- Expand dataset with **more years of data** for better climate trend analysis.
- Implement **real-time weather anomaly detection** using live data streams.




