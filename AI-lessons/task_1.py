from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[15, 1], [9, 2], [30, 0], [-2, 4], [5, 2]])
y = np.array([2, 1, 3, 1, 1])

model = LinearRegression()
model.fit(X, y)

temperature = float(input("Enter temperature in °C: "))
humidity = float(input("Enter humidity level: "))

new_weather = np.array([[temperature, humidity]])
prediction = model.predict(new_weather)[0]

if prediction < 1.5:
    print("Weather is cold")
elif 1.5 <= prediction < 2.5:
    print("Weather is warm")
else:
    print("Weather is hot")