import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.array([
    [0],
    [1]
])

y = np.array([0, 1])  # 0-sad, 1-happy

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

mood = input("How are you feeling today? (Happy/Sad): ").strip().lower()

mood_value = 1 if mood == "happy" else 0

predicted_advice = model.predict([[mood_value]])

if predicted_advice == 0:
    advice = "How about taking a break and doing something fun? Keep your chin up!"
elif predicted_advice == 1:
    advice = "Great to hear you're happy! Keep spreading those good vibes and smile more!"

print(f"\nBased on your mood ({mood.capitalize()})")
print(f"Here's some advice to make you smile: {advice}")