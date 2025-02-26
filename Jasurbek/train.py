import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


emails = []
with open('turkmen_text.txt', 'r') as file:
    sentences = file.readlines()
    emails = [sentence.strip() for sentence in sentences]

labels = []
with open('labels.txt', 'r') as file:
    sentences = file.readlines()
    labels = [int(sentence) for sentence in sentences]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()
print(f"Shape of X: {X.shape}")

np.random.seed(1)
input_size = X.shape[1]
print('Input size:', input_size)
hidden_size = 4
output_size = 1

weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)


def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return hidden_layer_output, predicted_output


def backpropagation(X, y, hidden_output, predicted_output, weights_hidden_output, weights_input_hidden,
                    learning_rate=0.1):
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_hidden_layer = d_predicted_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

    return weights_hidden_output, weights_input_hidden


y = np.array(labels).reshape(-1, 1)
print(f"Shape of y: {y.shape}")

for epoch in range(10000):
    hidden_output, predicted_output = forward_propagation(X, weights_input_hidden, weights_hidden_output)
    weights_hidden_output, weights_input_hidden = backpropagation(X, y, hidden_output, predicted_output,
                                                                  weights_hidden_output, weights_input_hidden)

s = ''
while s != 'end':
    s = input('Test sözlemi giriziň: ')
    if s.strip():
        s_array = [s]
        test_input = vectorizer.transform(s_array).toarray()
        _, prediction = forward_propagation(test_input, weights_input_hidden, weights_hidden_output)
        if prediction >= 0.5:
            print("Spam")
        else:
            print("Not spam")
