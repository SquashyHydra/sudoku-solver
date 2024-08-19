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
    # Find the first empty cell
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

def create_puzzle(solution, num_cells_to_remove, max_attempts=20):
    for attempt in range(max_attempts):
        puzzle = solution.copy()
        cells = list((i, j) for i in range(9) for j in range(9))
        random.shuffle(cells)
        
        # Store removed cells
        removed_cells = set()
        
        for i in range(num_cells_to_remove):
            row, col = cells[i]
            removed_cells.add((row, col))
            puzzle[row][col] = 0
        
        # Debug information
        print(f"Attempt {attempt + 1}: Removed {num_cells_to_remove} cells")
        
        # Ensure the puzzle has a solution count between 2 and 6
        solution_count = count_solutions(puzzle)
        print(f"Solution count: {solution_count}")
        if solution_count <= 6:
            return puzzle
        
    raise Exception("Failed to create a puzzle with the desired number of solutions after several attempts")

def count_solutions(puzzle):
    solutions_count = [0]  # Use a mutable object to count solutions

    def count(board):
        if solutions_count[0] > 6:  # Early exit if more than 6 solutions is found
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

def generate_single_puzzle():
    print("Generating Solution")
    solution = generate_solution()
    num_cells_to_remove = random.randint(40, 60)
    print("Creating Puzzle")
    puzzle = create_puzzle(solution, num_cells_to_remove)
    return puzzle, solution

def save_sudoku_data(filename, num_puzzles=100):
    counter = 0
    puzzles = []
    solutions = []
    
    try:
        with Pool(processes=cpu_count()) as pool:
            results = [pool.apply_async(generate_single_puzzle) for _ in range(num_puzzles)]
            results = [r.get() for r in results]  # Wait for all results
            
        puzzles, solutions = zip(*results)
        counter += len(puzzles)
        print(f"Generated {counter} puzzles", end="\r", flush=True)
        
    except KeyboardInterrupt:
        print(f"\nGeneration stopped. Total puzzles created: {counter}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    finally:
        data = {
            'puzzles': [p.flatten().tolist() for p in puzzles],
            'solutions': [s.flatten().tolist() for s in solutions]
        }
        
        with open(filename, 'w') as file:
            json.dump(data, file)
        print(f"\nData saved to {filename}")

def continuously_generate_sudoku_puzzles(filename):
    counter = 0
    puzzles = []
    solutions = []

    try:
        while True:
            try:
                with Pool(processes=cpu_count()) as pool:
                    results = [pool.apply_async(generate_single_puzzle) for _ in range(10)]
                    results = [r.get() for r in results]  # Wait for all results

                new_puzzles, new_solutions = zip(*results)
                puzzles.extend(new_puzzles)
                solutions.extend(new_solutions)
                
                counter += len(new_puzzles)
                print(f"Generated {counter} puzzles", end="\r", flush=True)
                
                # Optionally save periodically to avoid large memory usage
                if counter % 100 == 0:
                    data = {
                        'puzzles': [p.flatten().tolist() for p in puzzles],
                        'solutions': [s.flatten().tolist() for s in solutions]
                    }
                    with open(filename, 'w') as file:
                        json.dump(data, file)
                    print(f"\nSaved data at count {counter}")
                    
            except KeyboardInterrupt:
                print(f"\nGeneration stopped. Total puzzles created: {counter}")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                
    finally:
        # Save remaining puzzles and solutions before exiting
        data = {
            'puzzles': [p.flatten().tolist() for p in puzzles],
            'solutions': [s.flatten().tolist() for s in solutions]
        }
        with open(filename, 'w') as file:
            json.dump(data, file)
        print(f"\nFinal data saved. Total puzzles created: {counter}")

# Example Usage
if __name__ == "__main__":
    save_sudoku_data('sudoku_data.json', num_puzzles=1000)
    # continuously_generate_sudoku_puzzles('sudoku_data.json')