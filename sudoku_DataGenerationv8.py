import os, json, time, random
import numpy as np
import multiprocessing as mp

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
    return puzzle

def has_unique_solution(puzzle):
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

    return count_solutions(puzzle) == 1

def generate_single_puzzle(used_puzzles, lock, puzzle_counter):
    while True:
        solution = generate_solution()
        if solution is not False:
            num_cells_to_remove = random.randint(40, 60)
            puzzle = create_puzzle(solution, num_cells_to_remove)
            if has_unique_solution(puzzle):
                puzzle_str = np.array2string(puzzle, separator=',')
                with lock:
                    if puzzle_str not in used_puzzles:
                        used_puzzles.append(puzzle_str)
                        puzzle_counter.value += 1
                        return puzzle, solution

def worker(used_puzzles, lock, return_dict, puzzle_counter):
    puzzles = []
    solutions = []
    while puzzle_counter.value < 1000:  # Stop when 1000 puzzles are generated
        try:
            puzzle, solution = generate_single_puzzle(used_puzzles, lock, puzzle_counter)
            if puzzle is not None:
                puzzles.append(puzzle.tolist())
                solutions.append(solution.tolist())
        except Exception as e:
            print(f"Error in worker {mp.current_process().name}: {e}")
            break
    return_dict[mp.current_process().name] = (puzzles, solutions)

def save_sudoku_puzzles(filename, num_puzzles=1000):
    # Load existing data if it exists
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            existing_data = json.load(file)
            puzzles = existing_data['puzzles']
            solutions = existing_data['solutions']
    else:
        puzzles = []
        solutions = []

    manager = mp.Manager()
    used_puzzles = manager.list([np.array2string(np.array(p), separator=',') for p in puzzles])
    lock = manager.Lock()
    return_dict = manager.dict()
    puzzle_counter = manager.Value('i', len(puzzles))  # Initialize with the count of existing puzzles
    
    processes = []
    num_workers = mp.cpu_count()  # Number of processes to run in parallel
    
    # Create worker processes
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(used_puzzles, lock, return_dict, puzzle_counter))
        processes.append(p)
        p.start()
    
    # Monitor the progress
    while any(p.is_alive() for p in processes):
        with lock:  # Use the lock when printing
            print(f"Number of puzzles generated: {puzzle_counter.value}", end="\r", flush=True)
        time.sleep(1)  # Update every second
    
    # Wait for all worker processes to finish
    for p in processes:
        p.join()
    
    # Aggregate results
    for key in return_dict:
        worker_puzzles, worker_solutions = return_dict[key]
        puzzles.extend(worker_puzzles)
        solutions.extend(worker_solutions)
    
    # Truncate the lists if they exceed the required number of puzzles
    puzzles = puzzles[:num_puzzles]
    solutions = solutions[:num_puzzles]

    # Save the puzzles and solutions to the file
    with open(filename, 'w') as file:
        json.dump({'puzzles': puzzles, 'solutions': solutions}, file)

if __name__ == "__main__":
    save_sudoku_puzzles('sudoku_data.json', num_puzzles=1000)  # Generate and save puzzles