import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
def load_sudoku_data(filename):
    with open(filename, 'r') as file:
        dataset = json.load(file)
    X = np.array(dataset['data'])
    y = np.array(dataset['labels'])
    return X, y

# Define the model
def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(81,)),
        Dense(128, activation='relu'),
        Dense(81, activation='sigmoid')  # Output layer for 81 cells
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model():
    X_train, y_train = load_sudoku_data('sudoku_datav1.json')
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    model.save('sudoku_ai_model.keras')  # Save the model

if __name__ == "__main__":
    train_model()