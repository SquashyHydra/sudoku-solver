import numpy as np
import json
import random

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
    empty = find_empty_location(board)
    if not empty:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve(board):
                return True
            board[row][col] = 0
    return False

def find_empty_location(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return (row, col)
    return None

def count_solutions(board):
    solutions_count = [0]
    
    def count(board):
        if solutions_count[0] > 1:
            return
        empty = find_empty_location(board)
        if not empty:
            solutions_count[0] += 1
            return
        row, col = empty
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                count(board)
                board[row][col] = 0
        return

    count(board.copy())
    return solutions_count[0]

def generate_solution():
    board = np.zeros((9, 9), dtype=int)
    nums = list(range(1, 10))
    
    for i in range(9):
        random.shuffle(nums)
        for j in range(9):
            for num in nums:
                if is_valid(board, i, j, num):
                    board[i][j] = num
                    break
    if not solve(board):
        return False

    return board

def create_puzzle(solution, num_cells_to_remove):
    puzzle = solution.copy()
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    
    for i in range(num_cells_to_remove):
        row, col = cells[i]
        puzzle[row][col] = 0
    
    # Ensure the puzzle has a unique solution and no empty cells
    if count_solutions(puzzle) != 1 or np.any(puzzle == 0):
        return False

    return puzzle

def generate_single_puzzle(used_puzzles):
    while True:
        solution = generate_solution()
        if solution is not False:
            num_cells_to_remove = random.randint(40, 60)
            puzzle = create_puzzle(solution, num_cells_to_remove)
            if puzzle is not False:
                puzzle_str = np.array2string(puzzle, separator=',')
                if puzzle_str not in used_puzzles:
                    used_puzzles.add(puzzle_str)
                    return puzzle, solution

def save_sudoku_puzzles(filename, num_puzzles=100):
    count = 0
    puzzles = []
    solutions = []
    used_puzzles = set()
    
    for _ in range(num_puzzles):
        try:
            puzzle, solution = generate_single_puzzle(used_puzzles)
            if puzzle is not None:
                puzzles.append(puzzle.tolist())
                solutions.append(solution.tolist())

                count += 1
                print(f"Sudoku Board Generated: {count}", end="\r", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Failed to generate puzzle: {e}")
            continue
        finally:
            with open(filename, 'w') as file:
                json.dump({'puzzles': puzzles, 'solutions': solutions}, file)

if __name__ == "__main__":
    save_sudoku_puzzles('sudoku_data.json', num_puzzles=1000)  # Generate and save puzzles