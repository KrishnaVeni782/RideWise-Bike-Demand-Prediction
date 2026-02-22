# RideWise: Predicting Bike-Sharing Demand Based on Weather and Urban Events

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("🚲 RideWise Project Started...")

# Step 2: Load Dataset
data = pd.read_csv("bike_data.csv")

print("\nDataset Preview:")
print(data.head())

# Step 3: Define Features (Input) and Target (Output)
X = data[['temperature', 'humidity', 'windspeed', 'holiday', 'workingday']]
y = data['count']

# Step 4: Split Dataset (Training and Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Machine Learning Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Step 6: Predict Bike Demand
y_pred = model.predict(X_test)

# Step 7: Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Step 8: Sample Prediction (New Input)
print("\n🔮 Predicting Bike Demand for New Weather Data...")
sample_input = [[29, 65, 10, 0, 1]]  # temperature, humidity, windspeed, holiday, workingday
prediction = model.predict(sample_input)

print("Predicted Bike Demand:", int(prediction[0]), "bikes")

# Step 9: Visualization Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Bike Demand")
plt.ylabel("Predicted Bike Demand")
plt.title("RideWise: Actual vs Predicted Bike Demand")
plt.show()