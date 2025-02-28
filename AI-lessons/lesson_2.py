from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([[50, 1], [70, 2], [100, 3], [120, 4], [150, 55]])
y = np.array([1500, 2000, 2500, 9000, 3500])

model = LinearRegression()

model.fit(x, y)
area = float(input("Enter your area in meters: "))
bedroom = float(input("Enter your bedroom: "))

new_house = np.array([[area, bedroom]])

prediction = model.predict(new_house)
print(f"your results: {prediction[0]})")
