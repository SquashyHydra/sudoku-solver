import tensorflow as tf
import numpy as np
import json

def one_hot_encode(solutions):
    # Convert solutions to one-hot encoded format
    # solutions shape: (num_samples, 81)
    num_samples, num_cells = solutions.shape
    num_classes = 9
    y_encoded = np.zeros((num_samples, num_cells, num_classes))
    
    for i in range(num_samples):
        for j in range(num_cells):
            value = solutions[i, j] - 1  # Sudoku values are 1-9; adjust to 0-8
            if value >= 0:  # Only encode non-zero values
                y_encoded[i, j, value] = 1
    
    return y_encoded

def load_sudoku_data(filename):
    with open(filename, 'r') as file:
        dataset = json.load(file)
    
    puzzles = np.array(dataset['puzzles'])
    solutions = np.array(dataset['solutions'])
    
    if puzzles.shape[1] != 81 or solutions.shape[1] != 81:
        raise ValueError("Puzzles and solutions must have shape (number of samples, 81)")
    
    # One-hot encode the solutions
    solutions_encoded = one_hot_encode(solutions)
    
    return puzzles, solutions_encoded

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
    
    model.save('sudoku_ai_modelv3.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()