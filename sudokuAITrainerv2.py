import tensorflow as tf
import numpy as np
import json

def load_sudoku_data(filename):
    with open(filename, 'r') as file:
        dataset = json.load(file)
    
    puzzles = np.array(dataset['puzzles'])
    solutions = np.array(dataset['solutions'])
    
    if puzzles.shape[1] != 81 or solutions.shape[1] != 81:
        raise ValueError("Puzzles and solutions must have shape (number of samples, 81)")
    
    return puzzles, solutions

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(81,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(81 * 9, activation='softmax')  # Output probabilities for each of 9 digits per cell
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X_train, y_train = load_sudoku_data('sudoku_datav1.json')
    
    model = build_model()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    
    model.save('sudoku_ai_modelv2.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()