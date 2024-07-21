import numpy as np
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import json

def fetch_data(url):
    """
    Fetches and preprocesses data from the given URL.
    
    Args:
        url (str): The URL to fetch data from.
    
    Returns:
        str: The preprocessed text data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\W+', ' ', text).lower()
        return text
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return ""

def tokenize(text):
    """
    Tokenizes the input text and builds a vocabulary.
    
    Args:
        text (str): The input text to tokenize.
    
    Returns:
        tuple: A tuple containing the list of words, word-to-index mapping, and index-to-word list.
    """
    words = text.split()
    word2index = defaultdict(lambda: len(word2index))
    index2word = []
    for word in words:
        if word not in word2index:
            index2word.append(word)
        word2index[word]
    return words, word2index, index2word

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, vocab_size, hidden_size):
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1 / self.input_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1 / self.hidden_size)
        
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    
    def feedforward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return output
    
    def backpropagate(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backpropagate(X, y, output, learning_rate)
            if epoch % 10 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

def train_model(url, total_epochs=1000, epoch_step=100):
    """
    Trains the neural network model using data from the given URL.
    
    Args:
        url (str): The URL to fetch training data from.
        total_epochs (int): Total number of epochs to train the model.
        epoch_step (int): Number of epochs to run in each iteration.
    
    Returns:
        tuple: The word-to-index mapping and index-to-word list.
    """
    text = fetch_data(url)
    words, word2index, index2word = tokenize(text)
    
    vocab_size = len(word2index)
    hidden_size = 50  # Increased hidden layer size for more complexity
    nn = NeuralNetwork(vocab_size, hidden_size)
    
    X = np.zeros((len(words), vocab_size))
    y = np.zeros((len(words), vocab_size))
    for i, word in enumerate(words):
        if i < len(words) - 1:
            X[i, word2index[word]] = 1
            y[i, word2index[words[i + 1]]] = 1
    
    current_epoch = 0
    while current_epoch < total_epochs:
        nn.train(X, y, epochs=epoch_step, learning_rate=0.01)
        current_epoch += epoch_step
        save_model(word2index, index2word, nn)
        print(f"Completed {current_epoch} epochs out of {total_epochs}")
    
    return word2index, index2word

def generate_text(word2index, index2word, nn, seed_word, length=50):
    """
    Generates text using the trained neural network model.
    
    Args:
        word2index (dict): The word-to-index mapping.
        index2word (list): The index-to-word list.
        nn (NeuralNetwork): The trained neural network.
        seed_word (str): The seed word to start text generation.
        length (int): The length of the generated text.
    
    Returns:
        str: The generated text.
    """
    vocab_size = len(word2index)
    
    generated_text = seed_word
    current_word = seed_word
    
    for _ in range(length):
        X = np.zeros((1, vocab_size))
        if current_word in word2index:
            X[0, word2index[current_word]] = 1
        else:
            break
        
        output = nn.feedforward(X)
        next_word_index = np.argmax(output)
        next_word = index2word[next_word_index]
        
        generated_text += " " + next_word
        current_word = next_word
    
    return generated_text

def save_model(word2index, index2word, nn, filename='model.json'):
    """
    Saves the model to a file.
    
    Args:
        word2index (dict): The word-to-index mapping.
        index2word (list): The index-to-word list.
        nn (NeuralNetwork): The trained neural network.
        filename (str): The filename to save the model.
    """
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
    """
    Loads the model from a file.
    
    Args:
        filename (str): The filename to load the model from.
    
    Returns:
        tuple: The word-to-index mapping, index-to-word list, and neural network.
    """
    try:
        with open(filename, 'r') as f:
            model_data = json.load(f)
        word2index = model_data['word2index']
        index2word = model_data['index2word']
        vocab_size = len(word2index)
        hidden_size = len(model_data['bias_hidden'][0])
        nn = NeuralNetwork(vocab_size, hidden_size)
        nn.weights_input_hidden = np.array(model_data['weights_input_hidden'])
        nn.weights_hidden_output = np.array(model_data['weights_hidden_output'])
        nn.bias_hidden = np.array(model_data['bias_hidden'])
        nn.bias_output = np.array(model_data['bias_output'])
        return word2index, index2word, nn
    except FileNotFoundError:
        print("Model file not found.")
        return None, None, None
