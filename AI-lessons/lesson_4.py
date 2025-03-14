from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

questions = ["hello", "hi", "how are you", "your name", "bye"]
responses = ["Hello!", "Hi there!", "I'm good! How about you?", "I'm chatbot", "Goodbye!"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(questions)

model = LogisticRegression()
model.fit(X_train, responses)


def chatbot_reply(text):
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]


print("Chatbot: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ").lower()
    if user_input == "bye":
        print("Chatbot: Goodbye!")
        break
    print("Chatbot: ", chatbot_reply(user_input))
