import random
import numpy as np

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
