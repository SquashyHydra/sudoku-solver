import json, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

name_ai = "sudoku_ai_modelv4"
epochs = 40
batch_size = 32 # 32 or 64
learning_rate = 0.001

# early stop
patience = 100

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
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(81 * 9),
        tf.keras.layers.Reshape((81, 9)),
        tf.keras.layers.Activation('softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X, y = load_sudoku_data('sudoku_data.json')

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = build_model()

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f'best_{name_ai}.keras', save_best_only=True)
    ]
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    
    model.save(f'{name_ai}.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()