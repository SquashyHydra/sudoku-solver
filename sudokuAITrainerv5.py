import json
import numpy as np
import tensorflow as tf


name_ai = "sudoku_ai_modelv4"
batch_size = 32 # 32 or 64

def one_hot_encode(solutions):
    # Assuming solutions are of shape (number of samples, 81)
    num_classes = 9  # Since Sudoku numbers are from 1 to 9
    return np.eye(num_classes)[solutions - 1]  # One-hot encode and adjust indices

def load_sudoku_data(filename):
    with open(filename, 'r') as file:
        dataset = json.load(file)
    
    puzzles = np.array([np.array(puzzle).flatten() for puzzle in dataset['puzzles']])
    solutions = np.array([np.array(solution).flatten() for solution in dataset['solutions']])

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
        tf.keras.layers.Dense(81 * 9),  # Output shape will be (81 * 9)
        tf.keras.layers.Reshape((81, 9)),  # Reshape to (81, 9)
        tf.keras.layers.Activation('softmax')  # Apply softmax for probability distribution
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X_train, y_train = load_sudoku_data('sudoku_data.json')
    
    model = build_model()

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f'best_{name_ai}.keras', save_best_only=True)
    ]
    
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1, callbacks=callbacks)
    
    model.save(f'{name_ai}.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()