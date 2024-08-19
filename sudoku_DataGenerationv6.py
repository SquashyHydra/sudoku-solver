import numpy as np
import json
import random
from multiprocessing import Pool, cpu_count

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
            if board[row][col] == 0:  # Empty cell
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
    
    # Fill the diagonal 3x3 boxes to start
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for r in range(3):
            for c in range(3):
                board[i + r][i + c] = nums.pop()
    
    # Solve the board to fill it completely
    if not solve(board):
        raise Exception("Failed to generate a valid Sudoku board")
    
    return board

def create_puzzle(solution, num_cells_to_remove):
    puzzle = solution.copy()
    cells = list((i, j) for i in range(9) for j in range(9))
    random.shuffle(cells)
    
    # Store removed cells
    removed_cells = set()
    
    for i in range(num_cells_to_remove):
        row, col = cells[i]
        removed_cells.add((row, col))
        puzzle[row][col] = 0

    # Ensure the puzzle has a unique solution
    if not has_unique_solution(puzzle, removed_cells):
        return create_puzzle(solution, num_cells_to_remove)  # Retry if no unique solution
    
    return puzzle

def has_unique_solution(puzzle, removed_cells):
    def count_solutions(board):
        solutions_count = [0]  # Use a mutable object to count solutions

        def count(board):
            if solutions_count[0] > 1:  # Early exit if more than one solution is found
                return
            for row in range(9):
                for col in range(9):
                    if board[row][col] == 0:
                        for num in range(1, 10):
                            if is_valid(board, row, col, num):
                                board[row][col] = num
                                count(board)
                                board[row][col] = 0
                        return
            solutions_count[0] += 1

        count(puzzle.copy())
        return solutions_count[0]

    return count_solutions(puzzle) == 1

def generate_single_puzzle():
    solution = generate_solution()
    num_cells_to_remove = random.randint(40, 60)
    puzzle = create_puzzle(solution, num_cells_to_remove)
    return puzzle, solution

def save_sudoku_data(filename, num_puzzles=100):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(generate_single_puzzle, range(num_puzzles))

    puzzles, solutions = zip(*results)

    data = {
        'puzzles': [p.flatten().tolist() for p in puzzles],
        'solutions': [s.flatten().tolist() for s in solutions]
    }
    
    with open(filename, 'w') as file:
        json.dump(data, file)

def continuously_generate_sudoku_puzzles(filename):
    counter = 0
    puzzles = []
    solutions = []
    
    try:
        while True:
            try:
                solution = generate_solution()
                num_cells_to_remove = random.randint(40, 60)
                puzzle = create_puzzle(solution, num_cells_to_remove)
                counter += 1
                print(f"Generated puzzle number: {counter}", end="\r", flush=True)
                puzzles.append(puzzle.flatten().tolist())
                solutions.append(solution.flatten().tolist())
            except KeyboardInterrupt:
                print(f"\nGeneration stopped. Total puzzles created: {counter}")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                # Optionally continue generating if an error occurs
    finally:
        # Save data to file
        data = {
            'puzzles': puzzles,
            'solutions': solutions
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

# Example Usage
if __name__ == "__main__":
    #save_sudoku_data('sudoku_datav4.json', num_puzzles=1000)  # Generate and save puzzles and solutions
    continuously_generate_sudoku_puzzles('sudoku_datav5.json')