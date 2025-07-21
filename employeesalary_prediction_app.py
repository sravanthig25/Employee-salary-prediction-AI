import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# Sidebar for user input
st.sidebar.header("Employee Salary Prediction")
st.sidebar.write("Enter employee details to predict salary")

# Input fields in sidebar
years_experience = st.sidebar.slider("Years of Experience", 0.0, 20.0, 5.0, step=0.1)
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
age = st.sidebar.slider("Age", 18, 65, 30, step=1)
job_role = st.sidebar.selectbox("Job Role", ["Junior Developer", "Senior Developer", "Manager", "Director"])

# Encode education level and job role
education_mapping = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
job_role_mapping = {"Junior Developer": 1, "Senior Developer": 2, "Manager": 3, "Director": 4}
education_encoded = education_mapping[education_level]
job_role_encoded = job_role_mapping[job_role]

# Main content
st.title("Employee Salary Prediction Using Linear Regression")
st.write("This app predicts employee salaries based on experience, education, age, and job role.")

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
data = {"YearsExperience": np.random.uniform(0, 20, n_samples),
    "EducationLevel": np.random.choice([1, 2, 3, 4], n_samples),
    "Age": np.random.randint(18, 65, n_samples),
    "JobRole": np.random.choice([1, 2, 3, 4], n_samples),
    "Salary": 30000 + 5000 * np.random.uniform(0, 20, n_samples) +
             10000 * np.random.choice([1, 2, 3, 4], n_samples) +
             500 * np.random.randint(18, 65, n_samples) +
             15000 * np.random.choice([1, 2, 3, 4], n_samples) +
             np.random.normal(0, 5000, n_samples)
}
df = pd.DataFrame(data)

# Display dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Data preprocessing
X = df[["YearsExperience", "EducationLevel", "Age", "JobRole"]]
y = df["Salary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": ["YearsExperience", "EducationLevel", "Age", "JobRole"],
    "Coefficient": model.coef_
})
st.subheader("Feature Importance")
st.write(feature_importance)

# Plot feature importance
fig, ax = plt.subplots()
sns.barplot(x="Coefficient", y="Feature", data=feature_importance, ax=ax)
st.pyplot(fig)

# Prediction for user input
input_data = np.array([[years_experience, education_encoded, age, job_role_encoded]])
input_scaled = scaler.transform(input_data)
predicted_salary = model.predict(input_scaled)[0]

st.subheader("Salary Prediction")
st.write(f"Predicted Salary: ${predicted_salary:,.2f}")

# Visualization of actual vs predicted salaries
st.subheader("Actual vs Predicted Salaries")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
st.pyplot(fig)