import numpy as np
import tensorflow as tf

class SudokuAI:
    def __init__(self, grid):
        self.grid = grid
        self.model = tf.keras.models.load_model('sudoku_ai_modelv2.keras')  # Load the trained model

    def predict_next_move(self, grid):
        flattened_grid = np.array(grid).flatten().reshape(1, -1)
        prediction = self.model.predict(flattened_grid)
        # Reshape the prediction to match the Sudoku grid dimensions
        return prediction.reshape((9, 9))

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

# Example Sudoku puzzle
sudoku_puzzle = [
    [1, 3, 0, 0, 5, 0, 0, 0, 0],
    [5, 0, 0, 7, 0, 0, 0, 0, 2],
    [0, 0, 2, 0, 0, 9, 0, 0, 0],
    [9, 1, 0, 0, 0, 0, 0, 8, 7],
    [0, 0, 7, 0, 6, 0, 1, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [3, 4, 0, 0, 0, 0, 0, 6, 3],
    [0, 8, 0, 0, 0, 0, 0, 5, 0]
]

# Instantiate the SudokuAI with the puzzle and predict
ai = SudokuAI(sudoku_puzzle)
predicted_grid = ai.predict_next_move(sudoku_puzzle)

print("Predicted Grid:")
print(predicted_grid)