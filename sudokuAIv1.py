from sklearn.ensemble import RandomForestClassifier
import numpy as np

class SudokuAI:
    def __init__(self, grid):
        self.grid = grid
        self.model = self.train_model()

    def train_model(self):
        # Hypothetical training data for demonstration purposes
        X_train = np.random.randint(1, 10, (1000, 81))  # Random Sudoku grids
        y_train = np.random.randint(1, 10, (1000,))  # Random correct next moves
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model

    def predict_next_move(self, grid):
        # Predicts the best next move based on the current grid state
        flattened_grid = np.array(grid).flatten().reshape(1, -1)
        return self.model.predict(flattened_grid)

    def find_empty_cell(self):
        for i in range(9):
            for j in range(9):
                if self.grid[i][j] == 0:
                    return (i, j)
        return None

    def used_in_row(self, num, row):
        return num in self.grid[row]

    def used_in_column(self, num, col):
        return num in [self.grid[i][col] for i in range(9)]

    def used_in_box(self, num, row, col):
        box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
        return num in [self.grid[i][j] for i in range(box_start_row, box_start_row + 3) for j in range(box_start_col, box_start_col + 3)]

    def solve(self):
        find = self.find_empty_cell()
        if not find:
            return True
        
        row, col = find
        
        for num in range(1, 10):
            if not (self.used_in_row(num, row) or self.used_in_column(num, col) or self.used_in_box(num, row, col)):
                self.grid[row][col] = num
                self.print_grid_unsolved()
                if self.solve():
                    return True
                self.grid[row][col] = 0

        return False
        
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

    def print_grid_unsolved(self):
        printing = "Solving:\n"
        for i in range(9): 
            if i % 3 == 0 and i != 0: 
                printing += f"-----------------------\n"
              
            for j in range(9): 
                if j % 3 == 0 and j != 0:
                    printing += " | "
              
                if self.grid[i][j] == 0:
                    printing += "X "
                else: 
                    printing += f"{self.grid[i][j]} "
            printing += '\n'
        printing += '\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F'
        print(f"{printing}", end="\r", flush=True)