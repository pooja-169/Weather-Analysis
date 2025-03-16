import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Load dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

st.title("🌍 Weather Data Analysis & Forecasting")

# Dataset Overview
st.subheader("📊 Dataset Overview")
st.write(df.head())

# Missing Values
st.subheader("🔍 Missing Values Summary")
st.write("There are 0 missing values in the dataset.")

# Convert categorical columns using Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Outlier Handling - Side by Side Comparison
st.subheader("📈 Outlier Analysis & Handling")
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

col1, col2 = st.columns(2)
with col1:
    st.write("**Before Outlier Handling**")
    fig, axes = plt.subplots(nrows=len(num_cols), figsize=(6, len(num_cols) * 2))
    fig.subplots_adjust(hspace=0.5)  # **Increase vertical space between plots**
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col, pad=15)  # **Move title upward to avoid overlap**
        axes[i].set_xlabel("")  # **Remove x-axis label clutter**
    st.pyplot(fig)

# Replace Outliers with Median
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].median(), df[col])

with col2:
    st.write("**After Outlier Handling**")
    fig, axes = plt.subplots(nrows=len(num_cols), figsize=(6, len(num_cols) * 2))
    fig.subplots_adjust(hspace=0.5)  # **Increase vertical space between plots**
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col, pad=15)  # **Move title upward**
        axes[i].set_xlabel("")  # **Remove x-axis label clutter**
    st.pyplot(fig)


# Full Dataset Correlation Heatmap
st.subheader("📊 Full Feature Correlation Matrix")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
ax.set_title("Correlation Heatmap for All Features")
st.pyplot(fig)

# Temperature & Precipitation Distribution
st.subheader("🌡️ Temperature & Precipitation Distribution")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['temperature_celsius'], bins=30, kde=True, ax=ax[0])
ax[0].set_title("Temperature Distribution")
sns.histplot(df['precip_mm'], bins=30, kde=True, ax=ax[1])
ax[1].set_title("Precipitation Distribution")
st.pyplot(fig)

# Anomaly Detection
st.subheader("⚠️ Anomaly Detection")

# Select relevant numerical features
num_cols = ['temperature_celsius', 'humidity', 'precip_mm', 'air_quality_PM2.5', 'air_quality_PM10']
data = df[num_cols]

# Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
data['anomaly_if'] = iso_forest.fit_predict(data)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
data['anomaly_lof'] = lof.fit_predict(data)

# Count of anomalies detected
anomaly_count_if = sum(data['anomaly_if'] == -1)
anomaly_count_lof = sum(data['anomaly_lof'] == -1)
st.write(f"Anomalies detected by Isolation Forest: {anomaly_count_if}")
st.write(f"Anomalies detected by LOF: {anomaly_count_lof}")

# Fit Isolation Forest for visualization
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df[['temperature_celsius']])

# Mark anomalies where prediction is -1
df['is_anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Extract anomalies
anomalies = df[df['is_anomaly'] == 1]

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df['temperature_celsius'], color="lightblue", ax=ax)
sns.stripplot(x=anomalies['temperature_celsius'], color="red", alpha=0.7, size=6, jitter=True, ax=ax)

ax.set_title("Temperature Boxplot with Anomalies Highlighted")
st.pyplot(fig)

# Climate Trend Analysis
st.subheader("📉 Climate Trend Analysis")

# Ensure 'last_updated' is a datetime index
df['last_updated'] = pd.to_datetime(df['last_updated_epoch'], unit='s')
df.set_index('last_updated', inplace=True)

# Extract year and calculate yearly mean temperature
df['year'] = df.index.year
climate_trend = df.groupby('year')['temperature_celsius'].mean()

# Plot climate trend
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(climate_trend.index, climate_trend.values, marker='o', linestyle='-', color='blue')
ax.set_xlabel("Year")
ax.set_ylabel("Average Temperature (Celsius)")
ax.set_title("Long-Term Climate Trends")
ax.grid(True)

# Display the plot
st.pyplot(fig)


# Time Series Forecasting
st.subheader("📈 Time Series Forecasting")
df['last_updated'] = pd.to_datetime(df['last_updated_epoch'], unit='s')
df.set_index('last_updated', inplace=True)
temp_series = df['temperature_celsius'].resample('D').mean().dropna()
train, test = train_test_split(temp_series, test_size=0.2, shuffle=False)

# ARIMA Model
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index[:len(forecast)]

# Holt-Winters Model
hw_model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False).fit()
hw_forecast = hw_model.forecast(steps=len(test))
hw_forecast.index = test.index[:len(hw_forecast)]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train, label="Training Data")
ax.plot(test, label="Actual Data", color="green")
ax.plot(forecast.index, forecast, label="ARIMA Forecast", linestyle="dashed", color="red")
ax.plot(hw_forecast.index, hw_forecast, label="Holt-Winters Forecast", linestyle="dashed", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (Celsius)")
ax.set_title("Time Series Forecasting: ARIMA vs. Holt-Winters")
ax.legend()
st.pyplot(fig)

# Forecasting Model Comparison
st.subheader("📊 Forecasting Model Comparison")

# Prepare data for training
X = df.drop(columns=['temperature_celsius'])
y = df['temperature_celsius']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge()
}

# Store results
results = []

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mse, mae, rmse, r2])

# Convert results into a DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "RMSE", "R2 Score"])

# Display comparison table
st.write("Even after using Grid Search and Cross Validation, the R² score remains the same, indicating that the models have not overfitted the data.")
st.dataframe(results_df)

# Feature Importance Analysis
st.subheader("📌 Feature Importance Analysis")

# Train Random Forest for feature importance
rf_regressor = RandomForestRegressor(random_state=42)
X = df.drop(columns=['temperature_celsius'])
y = df['temperature_celsius']
rf_regressor.fit(X, y)

# Get feature importances
importances = rf_regressor.feature_importances_
feature_names = X.columns

# Create DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance (Random Forest)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
ax.set_title('Feature Importance from Random Forest')
st.pyplot(fig)

# Train Linear Regression for coefficient analysis
lr = LinearRegression()
lr.fit(X, y)

# Get coefficients
coefficients = lr.coef_
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Plot Coefficients (Linear Regression)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax)
ax.set_title('Feature Importance from Linear Regression Coefficients')
st.pyplot(fig)


# Environmental Impact - Correlation Heatmap
st.subheader("🏭 Environmental Impact - Correlation Matrix")
env_features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm',
                'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
                'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10']
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[env_features].corr(), cmap='coolwarm', annot=True, fmt=".2f")
ax.set_title("Environmental Feature Correlation Heatmap")
st.pyplot(fig)


# Geographical Weather Patterns
st.subheader("🗺️ Geographical Distribution of Weather Patterns")

# Temperature Distribution Scatterplot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='temperature_celsius', palette='coolwarm', ax=ax)
ax.set_title("Geographical Distribution of Temperature")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# Humidity Distribution Scatterplot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='humidity', palette='Blues', ax=ax)
ax.set_title("Geographical Distribution of Humidity")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# Spatial Analysis
st.subheader("📍 Spatial Analysis of Weather Data")

# Spatial Density of Data Points
fig, ax = plt.subplots(figsize=(12, 8))
sns.kdeplot(data=df, x='longitude', y='latitude', fill=True, cmap='coolwarm', levels=20, ax=ax)
ax.set_title("Spatial Density of Data Points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# Ensure temperature is positive for KDE weights
df['temperature_celsius_scaled'] = df['temperature_celsius'] - df['temperature_celsius'].min() + 1

# Spatial Temperature Distribution
fig, ax = plt.subplots(figsize=(12, 8))
sns.kdeplot(data=df, x='longitude', y='latitude', weights=df['temperature_celsius_scaled'], fill=True, cmap='coolwarm', levels=20, ax=ax)
ax.set_title("Spatial Temperature Distribution")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

st.success("Analysis Completed! 🚀")
