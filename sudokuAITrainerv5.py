import json, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
import datetime

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
    
    puzzles = puzzles - 1

    # One-hot encode the solutions
    solutions_encoded = one_hot_encode(solutions)

    return puzzles, solutions_encoded

def sudoku_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, [-1, 81, 9])

    # Standard categorical crossentropy loss
    base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Reshape predictions to [batch_size, 9, 9, 9] for easier validation
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred_reshaped = tf.reshape(y_pred, [-1, 9, 9])
    
    # Penalty for invalid rows
    row_penalty = tf.reduce_mean(tf.cast(tf.reduce_sum(tf.one_hot(y_pred_reshaped, 9), axis=2) != 1, tf.float32))
    
    # Penalty for invalid columns
    col_penalty = tf.reduce_mean(tf.cast(tf.reduce_sum(tf.one_hot(y_pred_reshaped, 9), axis=1) != 1, tf.float32))
    
    # Penalty for invalid 3x3 subgrids
    subgrid_penalty = 0
    for i in range(3):
        for j in range(3):
            subgrid = y_pred_reshaped[:, i*3:(i+1)*3, j*3:(j+1)*3]
            subgrid_penalty += tf.reduce_mean(tf.cast(tf.reduce_sum(tf.one_hot(subgrid, 9), axis=[1, 2]) != 1, tf.float32))
    
    total_penalty = row_penalty + col_penalty + subgrid_penalty
    
    # Combine the base loss with the penalties
    total_loss = base_loss + total_penalty
    
    return total_loss

def valid_sudoku_metric(y_true, y_pred):
    # Reshape y_pred to [batch_size, 9, 9] for easier manipulation
    y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1, 9, 9])
    
    # Define a function to compute penalties
    def penalty_grid(grid):
        # Ensure all digits from 1 to 9 are present exactly once in each row, column, and subgrid
        valid_rows = tf.reduce_all(tf.reduce_sum(tf.one_hot(grid, 9), axis=1) == 9, axis=1)
        valid_cols = tf.reduce_all(tf.reduce_sum(tf.one_hot(grid, 9), axis=0) == 9, axis=1)
        
        subgrid_penalty = 0
        for i in range(3):
            for j in range(3):
                subgrid = grid[i*3:(i+1)*3, j*3:(j+1)*3]
                valid_subgrid = tf.reduce_all(tf.reduce_sum(tf.one_hot(subgrid, 9), axis=[0, 1]) == 9)
                subgrid_penalty += tf.cast(tf.logical_not(valid_subgrid), tf.float32)
        
        # Compute row and column penalties
        row_penalty = tf.reduce_sum(tf.cast(tf.logical_not(valid_rows), tf.float32))
        col_penalty = tf.reduce_sum(tf.cast(tf.logical_not(valid_cols), tf.float32))
        
        # Total penalty
        total_penalty = row_penalty + col_penalty + subgrid_penalty
        return total_penalty
    
    # Apply penalty calculation to the entire batch
    penalties = tf.map_fn(penalty_grid, y_pred, dtype=tf.float32)
    mean_penalty = tf.reduce_mean(penalties)
    
    return mean_penalty

def model1():
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

    return model

def model2():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((9, 9, 1), input_shape=(81,)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(81 * 9, activation='softmax'),
        tf.keras.layers.Reshape((81, 9))
    ])

    return model

def build_model():
    model = model1()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=sudoku_loss, metrics=['accuracy', valid_sudoku_metric])
    return model

def train_and_save_model():
    x, y = load_sudoku_data('sudoku_data.json')

    # Split data into training and validation sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    model = build_model()

    # Define callbacks
    log_dir = "tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=tensorboard_callback)
    
    model.evaluate(x_test, y_test, verbose=2)

    model.save(f'{name_ai}.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()