import numpy as np
import tensorflow as tf

class SudokuAI:
    def __init__(self, grid):
        self.grid = grid
        try:
            self.model = tf.keras.models.load_model('sudoku_ai_modelv2.keras')
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_next_move(self, grid):
        flattened_grid = np.array(grid).flatten().reshape(1, -1)
        prediction = self.model.predict(flattened_grid)
        # Reshape prediction to 9x9 grid
        prediction_grid = prediction.reshape((9, 9, 9))
        # Get the most probable number for each cell
        predicted_numbers = np.argmax(prediction_grid, axis=2) + 1  # Adding 1 because labels start from 1
        return predicted_numbers

    def print_grid(self): 
        for i in range(9): 
            if i % 3 == 0 and i != 0: 
                print("-----------------------") 
              
            for j in range(9): 
                if j % 3 == 0 and j != 0: 
                    print(" | ", end = "") 
              
                if self.grid[i][j] == 0: 
                    print("X", end = " ") 
                else: 
                    print(self.grid[i][j], end = " ") 
            print()

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

# Check grid size
if len(sudoku_puzzle) != 9 or any(len(row) != 9 for row in sudoku_puzzle):
    raise ValueError("Sudoku grid must be 9x9")

# Instantiate and use the SudokuAI class
ai = SudokuAI(sudoku_puzzle)
predicted_grid = ai.predict_next_move(sudoku_puzzle)
print("Predicted Grid:")
print(predicted_grid)