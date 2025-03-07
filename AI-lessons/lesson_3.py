from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [25,5], [30,7], [27,6],
    [60,25], [65,30], [58,27],
    [28,5], [32,6], [63,28], [66,32]
])

model = KMeans(n_clusters=2, random_state=42)

prediction = model.fit_predict(X)

print(prediction)