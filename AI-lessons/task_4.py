from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([
    ["Mercedes", 2000], ["Toyota", 2010], ["Opel", 2015],
    ["Nissan", 2005], ["Lexus", 2001], ["Lada", 2012]
])
y = np.array([5000, 5000, 3500, 2500, 3000, 2300])

car_brands = X[:, 0].reshape(-1, 1)
years = X[:, 1].astype(int).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
encoded_brands = encoder.fit_transform(car_brands)
X_encoded = np.hstack([encoded_brands, years])

model = LinearRegression()
model.fit(X_encoded, y)

mark = input("(Example -> Mercedes, Toyota, Opel, Nissan, Lexus, Lada)\nEnter car's mark: ")
year = int(input("Enter year: "))

new_brand_encoded = encoder.transform(np.array([[mark]]))
new_data = np.hstack([new_brand_encoded, np.array([[year]])])

prediction = model.predict(new_data)
print(f"Car's price: {prediction[0]}")
