import numpy as np
import json

class Sudoku_DataGeneration:
    def __init__(self):
        self.data = []
        self.labels = []

    def generate_sudoku_data(self, n_samples=1000):
        for _ in range(n_samples):
            grid = self.generate_sudoku_grid()
            self.solve(grid, collect_data=True)
        self.save_data_to_file('sudoku_data.json')

    def generate_sudoku_grid(self):
        grid = np.zeros((9, 9), dtype=int)
        for _ in range(np.random.randint(10, 20)):
            row, col = np.random.randint(0, 9, size=2)
            num = np.random.randint(1, 10)
            if self.is_valid_move(grid, num, row, col):
                grid[row][col] = num
        return grid

    def is_valid_move(self, grid, num, row, col):
        return (num not in grid[row] and
                num not in grid[:, col] and
                num not in grid[row//3*3:(row//3+1)*3, col//3*3:(col//3+1)*3])

    def solve(self, grid, collect_data=False):
        empty_cell = self.find_empty_cell(grid)
        if not empty_cell:
            return True

        row, col = empty_cell
        for num in range(1, 10):
            if self.is_valid_move(grid, num, row, col):
                grid[row][col] = num
                if collect_data:
                    self.data.append(grid.flatten().tolist())
                    self.labels.append(num)
                if self.solve(grid, collect_data):
                    return True
                grid[row][col] = 0

        return False

    def find_empty_cell(self, grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def save_data_to_file(self, filename):
        with open(filename, 'w') as file:
            json.dump({'data': self.data, 'labels': self.labels}, file)

# Example Usage
if __name__ == "__main__":
    generator = Sudoku_DataGeneration()
    generator.generate_sudoku_data(n_samples=10000)  # Generate synthetic data and save to file