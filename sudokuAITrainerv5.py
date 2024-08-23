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
    """
    Custom metric for checking the validity of Sudoku solutions.
    """
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert predicted probabilities to the most likely class (1-9)
    y_true = tf.argmax(y_true, axis=-1)  # Same for true labels

    # Reshape predictions to (batch_size, 9, 9)
    y_pred = tf.reshape(y_pred, [-1, 9, 9])
    
    def check_valid_entries(tensor):
        # Ensure tensor has shape [batch_size, 9] (valid entries across rows or columns)
        return tf.reduce_all(tf.reduce_sum(tf.one_hot(tensor, 9), axis=1) == 1, axis=1)
    
    # Validate rows
    valid_rows = check_valid_entries(y_pred)  # Expected shape: [batch_size]
    valid_rows = tf.expand_dims(valid_rows, axis=1)  # Shape: [batch_size, 1]
    
    # Validate columns
    y_pred_transposed = tf.transpose(y_pred, perm=[0, 2, 1])  # Transpose to validate columns
    valid_cols = check_valid_entries(y_pred_transposed)  # Expected shape: [batch_size]
    valid_cols = tf.expand_dims(valid_cols, axis=1)  # Shape: [batch_size, 1]
    
    # Validate 3x3 subgrids
    def validate_subgrids(grid):
        batch_size = tf.shape(grid)[0]
        subgrid_validities = []
        for i in range(3):
            for j in range(3):
                subgrid = grid[:, i*3:(i+1)*3, j*3:(j+1)*3]
                subgrid_flattened = tf.reshape(subgrid, [batch_size, -1])
                valid_subgrid = check_valid_entries(subgrid_flattened)
                subgrid_validities.append(valid_subgrid)
        
        # Stack subgrid validities and reduce them
        valid_subgrids = tf.stack(subgrid_validities, axis=1)  # Shape: [batch_size, 9]
        valid_subgrids = tf.reduce_all(valid_subgrids, axis=1)  # Shape: [batch_size]
        valid_subgrids = tf.expand_dims(valid_subgrids, axis=1)  # Shape: [batch_size, 1]
        return valid_subgrids
    
    valid_subgrids = validate_subgrids(y_pred)

    # Debugging: Check shapes before stacking
    print(f"valid_rows shape: {valid_rows.shape}")
    print(f"valid_cols shape: {valid_cols.shape}")
    print(f"valid_subgrids shape: {valid_subgrids.shape}")

    # Ensure all tensors have the shape [batch_size, 1]
    valid_rows = tf.squeeze(valid_rows, axis=1)  # Shape: [batch_size]
    valid_cols = tf.squeeze(valid_cols, axis=1)  # Shape: [batch_size]
    valid_subgrids = tf.squeeze(valid_subgrids, axis=1)  # Shape: [batch_size]
    
    # Stack tensors and reduce across axis 1
    valid_sudoku = tf.reduce_all(tf.stack([valid_rows, valid_cols, valid_subgrids], axis=1), axis=1)
    
    return tf.reduce_mean(tf.cast(valid_sudoku, tf.float32))
    
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

    model.compile(optimizer=optimizer, loss=sudoku_loss, metrics=['accuracy', valid_sudoku_metric])
    return model

def train_and_save_model():
    X, y = load_sudoku_data('sudoku_data.json')

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = build_model()

    # Define callbacks
    log_dir = "tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=tensorboard_callback)
    
    model.save(f'{name_ai}.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()