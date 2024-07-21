import requests
import numpy as np
import json
import re

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, inputs):
        self.hidden = np.tanh(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        return self.softmax(output)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def backward(self, inputs, targets, outputs, learning_rate):
        output_error = outputs - targets
        hidden_error = output_error.dot(self.weights_hidden_output.T) * (1 - self.hidden ** 2)

        self.weights_hidden_output -= self.hidden.T.dot(output_error) * learning_rate
        self.bias_output -= output_error.mean(axis=0) * learning_rate
        self.weights_input_hidden -= inputs.T.dot(hidden_error) * learning_rate
        self.bias_hidden -= hidden_error.mean(axis=0) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            self.backward(inputs, targets, outputs, learning_rate)

def fetch_data(url):
    response = requests.get(url)
    return response.text

def tokenize(data):
    words = re.findall(r'\b\w+\b', data.lower())
    word2index = {word: i for i, word in enumerate(set(words))}
    index2word = {i: word for word, i in word2index.items()}
    tokens = [word2index[word] for word in words]
    return tokens, word2index, index2word

def prepare_data(tokens, vocab_size, sequence_length=10):
    X = []
    y = []
    for i in range(len(tokens) - sequence_length):
        X.append(tokens[i:i + sequence_length])
        y.append(tokens[i + sequence_length])
    X = np.eye(vocab_size)[np.array(X)]
    y = np.eye(vocab_size)[np.array(y)]
    return X, y

def train_model(url, epochs, sequence_length=10, learning_rate=0.01):
    data = fetch_data(url)
    tokens, word2index, index2word = tokenize(data)
    vocab_size = len(word2index)
    X, y = prepare_data(tokens, vocab_size, sequence_length)

    nn = NeuralNetwork(vocab_size, 128, vocab_size)
    nn.train(X, y, epochs, learning_rate)

    save_model(word2index, index2word, nn)
    return nn, word2index, index2word

def generate_text(seed_text, length, word2index, index2word, nn, sequence_length=10):
    generated_text = seed_text
    current_sequence = [word2index[word] for word in seed_text.split()]

    for _ in range(length):
        X = np.eye(len(word2index))[current_sequence[-sequence_length:]]
        X = X.reshape(1, sequence_length, len(word2index))
        output = nn.forward(X)
        next_word_index = np.argmax(output)
        next_word = index2word[next_word_index]
        generated_text += " " + next_word
        current_sequence.append(next_word_index)

    return generated_text

def save_model(word2index, index2word, nn, filename='model.json'):
    model_data = {
        'word2index': word2index,
        'index2word': index2word,
        'weights_input_hidden': nn.weights_input_hidden.tolist(),
        'weights_hidden_output': nn.weights_hidden_output.tolist(),
        'bias_hidden': nn.bias_hidden.tolist(),
        'bias_output': nn.bias_output.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(model_data, f)

def load_model(filename='model.json'):
    try:
        with open(filename, 'r') as f:
            model_data = json.load(f)
        word2index = model_data['word2index']
        index2word = model_data['index2word']
        vocab_size = len(word2index)
        hidden_size = len(model_data['bias_hidden'][0])
        nn = NeuralNetwork(vocab_size, hidden_size, vocab_size)
        nn.weights_input_hidden = np.array(model_data['weights_input_hidden'])
        nn.weights_hidden_output = np.array(model_data['weights_hidden_output'])
        nn.bias_hidden = np.array(model_data['bias_hidden'])
        nn.bias_output = np.array(model_data['bias_output'])
        return word2index, index2word, nn
    except FileNotFoundError:
        print("Model file not found.")
        return None, None, None