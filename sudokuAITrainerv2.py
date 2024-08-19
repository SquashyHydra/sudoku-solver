import tensorflow as tf
import numpy as np
import json

def load_sudoku_data(filename):
    with open(filename, 'r') as file:
        dataset = json.load(file)
    
    puzzles = np.array(dataset['puzzles'])
    solutions = np.array(dataset['solutions'])
    
    return puzzles, solutions

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(81,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(81, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_and_save_model():
    X_train, y_train = load_sudoku_data('sudoku_datav1.json')
    
    model = build_model()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    
    model.save('sudoku_solver_modelv2.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()