import random
import numpy as np
import json

def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[box_row:box_row + 3, box_col:box_col + 3]:
        return False
    return True

def solve(board):
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

def generate_solution():
    board = np.zeros((9, 9), dtype=int)
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for r in range(3):
            for c in range(3):
                board[i + r][i + c] = nums.pop()
    if not solve(board):
        raise Exception("Failed to generate a valid Sudoku board")
    return board

def create_puzzle(solution, num_cells_to_remove):
    puzzle = solution.copy()
    cells = list((i, j) for i in range(9) for j in range(9))
    random.shuffle(cells)
    for i in range(num_cells_to_remove):
        row, col = cells[i]
        puzzle[row][col] = 0
    return puzzle

def save_sudoku_data(filename, num_puzzles=100):
    puzzles = []
    solutions = []
    for _ in range(num_puzzles):
        solution = generate_solution()
        num_cells_to_remove = random.randint(40, 60)
        puzzle = create_puzzle(solution, num_cells_to_remove)
        puzzles.append(puzzle.flatten().tolist())
        solutions.append(solution.flatten().tolist())
    
    data = {
        'puzzles': puzzles,
        'solutions': solutions
    }
    
    with open(filename, 'w') as file:
        json.dump(data, file)

# Example Usage
if __name__ == "__main__":
    save_sudoku_data('sudoku_datav2.json', num_puzzles=1000)  # Generate and save puzzles and solutions