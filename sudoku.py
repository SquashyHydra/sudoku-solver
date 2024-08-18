import random
import numpy as np

class SudokuAI:
    def __init__(self, grid):
        self.grid = grid

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
                #print("-----------------------")
              
            for j in range(9): 
                if j % 3 == 0 and j != 0:
                    printing += " | "
                    #print(" | ", end = "") 
              
                if self.grid[i][j] == 0:
                    printing += "X "
                    #print("X", end = " ") 
                else: 
                    printing += f"{self.grid[i][j]} "
                    #print(self.grid[i][j], end = " ")
            printing += '\n'
        printing += '\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F'
        print(f"{printing}", end="\r", flush=True)

def create_sudoku():
    def is_valid(board, row, col, num):
        # Check if num is not in the current row
        if num in board[row]:
            return False
        # Check if num is not in the current column
        if num in board[:, col]:
            return False
        # Check if num is not in the current 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_row:box_row + 3, box_col:box_col + 3]:
            return False
        return True

    def solve(board):
        # Find the first empty cell
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if solve(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    board = np.zeros((9, 9), dtype=int)
    
    # Fill the diagonal boxes
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for r in range(3):
            for c in range(3):
                board[i + r][i + c] = nums.pop()
    
    # Solve the board
    if not solve(board):
        raise Exception("Failed to generate a valid Sudoku board")

    # Remove some numbers to create a puzzle
    num_cells_to_remove = random.randint(40, 60)
    cells = list((i, j) for i in range(9) for j in range(9))
    random.shuffle(cells)
    for i in range(num_cells_to_remove):
        row, col = cells[i]
        board[row][col] = 0

    return board

  
# Driver Code 
if __name__ == '__main__':
    def ask_grid():
        grid = [[3, 4, 5, 0, 7, 0, 0, 6, 2], 
                [7, 2, 0, 0, 4, 0, 0, 0, 0], 
                [0, 0, 1, 2, 5, 0, 0, 3, 0],  
                [0, 1, 0, 4, 0, 7, 5, 0, 6],  
                [6, 0, 0, 0, 1, 0, 0, 0, 4],  
                [0, 0, 0, 0, 8, 3, 0, 7, 9],  
                [4, 0, 0, 1, 3, 2, 0, 0, 0],  
                [0, 6, 2, 0, 9, 8, 0, 4, 3],  
                [5, 3, 0, 0, 0, 0, 0, 0, 1]] 
        
        rand_grid = input('Use Random Grid (Yes | No): ')
        if rand_grid.lower() in ["yes", "y"]:
            grid = create_sudoku()
        return grid

    def run_once():
        grid = ask_grid()
        sudoku = SudokuAI(grid)
        print("\nInitial Grid:")
        sudoku.print_grid()
        if not sudoku.solve():
            print("No solution exists")
        else:
            print("Solved Grid:")
            sudoku.print_grid()

    def multi_run():
        from time import sleep
        while True:
            try:
                grid = create_sudoku()
                sudoku = SudokuAI(grid)
                print("\nInitial Grid:")
                sudoku.print_grid()
                if not sudoku.solve():
                    print("No solution exists")
                else:
                    print("Solved Grid:")
                    sudoku.print_grid()
                    sleep(1)
            except KeyboardInterrupt:
                break

    multi_run()