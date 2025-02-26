from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([
    [25, 70, 7, 6],
    [30, 80, 5, 7],
    [22, 60, 8, 9],
    [40, 90, 4, 5],
    [35, 75, 6, 8],
])

y = np.array([24.2, 27.0, 20.0, 29.0, 20.0])

model = LinearRegression().fit(x,y)

print("Enter the following details:")

age = float(input("Age (in yerars): "))
weight = float(input("Weight: "))
exercise_level = float(input("Exercise level: "))
diet = float(input("Diet (1 to 10 range): "))

new_person = np.array([[age, weight, exercise_level, diet]])
predicted_bmi = model.predict(new_person)

print(f"Predicted BMI for the person: {predicted_bmi[0]:.2f}")
