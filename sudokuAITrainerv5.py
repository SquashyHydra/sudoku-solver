import json, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
import datetime

from sklearn.model_selection import train_test_split

name_ai = "sudoku_ai_modelv4.2"
EPOCH = 1000
batch_size = 32 # 32 or 64
learning_rate = 0.001

# early stop
patience = 100

CHECKPOINT="models/{name_ai}-{epoch:02d}-{loss:.2f}.keras"
LOGS='tmp/logs'

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

def model3():
    inputs = tf.keras.Input(shape=(81,))
    x = tf.keras.layers.Reshape((9, 9, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Add()([x, tf.keras.layers.Conv2D(32, (1, 1), padding='same')(x)])  # Residual connection
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(81 * 9, activation='softmax')(x)
    x = tf.keras.layers.Reshape((81, 9))(x)
    model = tf.keras.Model(inputs, x)

    return model

def model4():
    inputs = tf.keras.Input(shape=(81,))
    x = tf.keras.layers.Reshape((9, 9, 1))(inputs)
    
    # Encoder
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1_pool = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    
    x2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1_pool)
    x2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2_pool = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    
    # Bottleneck
    x3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x2_pool)
    x3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x3)
    
    # Decoder
    x2_up = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x3)
    x2_up = tf.keras.layers.Concatenate()([x2_up, x2])
    x2_up = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2_up)
    x2_up = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2_up)
    
    x1_up = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x2_up)
    x1_up = tf.keras.layers.Concatenate()([x1_up, x1])
    x1_up = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1_up)
    x1_up = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1_up)
    
    outputs = tf.keras.layers.Conv2D(81 * 9, (1, 1), activation='softmax')(x1_up)
    outputs = tf.keras.layers.Reshape((81, 9))(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def model5():
    inputs = tf.keras.Input(shape=(81,))
    x = tf.keras.layers.Reshape((9, 9))(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(81 * 9, activation='softmax')(x)
    outputs = tf.keras.layers.Reshape((81, 9))(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def model6():
    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(9,9,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=9, kernel_size=1, padding='same'),
    ])

    return model

def model7():
    from spektral.layers import GCNConv

    inputs = tf.keras.Input(shape=(81,))
    x = tf.keras.layers.Reshape((9, 9, 1))(inputs)
    
    # Convert the grid into a graph structure here
    # This requires a custom preprocessing step to convert the grid into graph data

    # Example architecture using GCN
    x = tf.keras.layers.Flatten()(x)  # Flatten grid to feed into GCN
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(81 * 9, activation='softmax')(x)
    outputs = tf.keras.layers.Reshape((81, 9))(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def build_model():
    model = model6()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

    model.summary()

    return model

def train_and_save_model():
    x, y = load_sudoku_data('sudoku_data.json')
    num_train = len(x)

    # Split data into training and validation sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    model = build_model()

    # Define callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT, monitor='sparse_categorical_accuracy', verbose=1, save_weights_only=False , save_best_only=True, mode='max')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='sparse_categorical_accuracy', factor=0.60, patience=3, min_lr=0.000001, verbose=1, mode='max')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOGS, histogram_freq=0, write_graph=True, write_images=True)
    
    callbacks_list = [checkpoint, tensorboard, reduce_lr]

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCH, steps_per_epoch=num_train//batch_size, batch_size=batch_size, callbacks=callbacks_list)
    
    model.evaluate(x_test, y_test, verbose=2)

    model.save(f'models/{name_ai}.keras')

# Example Usage
if __name__ == "__main__":
    train_and_save_model()