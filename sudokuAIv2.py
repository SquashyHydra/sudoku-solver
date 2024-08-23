import numpy as np
import tensorflow as tf
from sudokuAITrainerv5 import name_ai

class SudokuAI:
    def __init__(self, grid):
        self.grid = grid
        try:
            self.model = tf.keras.models.load_model(f'{name_ai}.keras')
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_next_move(self, grid):
        reshaped_grid = grid.reshape(1, -1)
        prediction = self.model.predict(reshaped_grid)
        print(f"Prediction shape: {prediction.shape}")  # Check the shape of prediction
        
        # The shape should be (1, 81 * 9) if the model predicts probabilities for 9 numbers for each cell
        prediction = prediction.reshape((9, 9, 9))  # Reshape to (9, 9, 9)
        
        # Get the most probable number for each cell
        predicted_numbers = np.argmax(prediction, axis=2) + 1  # Adding 1 because labels start from 1
        return predicted_numbers

def print_grid(grid):
    j = 0
    z = 0
    print(f"[", end="", flush=True)
    for i in grid:
        if j == 0 and z == 0:
            print(f"[", end="", flush=True)
        elif j == 0 and z != 0:
            print(f" [", end="", flush=True)

        if j != 8:
            print(f'{i}', end=" ", flush=True)
        elif j == 8:
            print(f'{i}', end="", flush=True)

        j += 1

        if j == 9 and not z == 8:
            j = 0
            z += 1
            print(f']', end="\n", flush=True)

        if z == 8 and j == 9:
            print(f"]", end='', flush=True)
    print(f"]", end="\n", flush=True)

sudoku_puzzle = [
    [0, 9, 0, 0, 0, 3, 0, 2, 0],
    [0, 7, 0, 0, 0, 6, 0, 0, 0],
    [0, 1, 0, 8, 0, 0, 0, 7, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 8, 0, 9, 0],
    [0, 2, 0, 0, 1, 0, 0, 5, 0],
    [0, 0, 4, 0, 0, 5, 0, 8, 0],  
    [0, 0, 2, 6, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 6]   # Added an empty row to make it 9x9
]

sudoku_puzzle = np.array(sudoku_puzzle).flatten()


# Instantiate and use the SudokuAI class
ai = SudokuAI(sudoku_puzzle)
predicted_grid = ai.predict_next_move(sudoku_puzzle)
print(f"Inital Grid:")
print_grid(sudoku_puzzle)
print(f"Predicted Grid:\n{predicted_grid}")