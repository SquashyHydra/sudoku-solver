from random import randint, shuffle
import json

# Initialize empty 9 by 9 grid
def create_empty_grid():
    return [[0 for _ in range(9)] for _ in range(9)]

# A function to check if the grid is full
def checkGrid(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return False
    return True

# A backtracking/recursive function to solve the grid
def solveGrid(grid):
    for i in range(81):
        row = i // 9
        col = i % 9
        if grid[row][col] == 0:
            for value in range(1, 10):
                if not (value in grid[row]):
                    if not value in [grid[r][col] for r in range(9)]:
                        square = [grid[r][col//3*3:col//3*3+3] for r in range(row//3*3, row//3*3+3)]
                        if not value in [num for sublist in square for num in sublist]:
                            grid[row][col] = value
                            if checkGrid(grid):
                                return True
                            else:
                                if solveGrid(grid):
                                    return True
            break
    grid[row][col] = 0

# A backtracking/recursive function to fill the grid
def fillGrid(grid):
    numberList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(81):
        row = i // 9
        col = i % 9
        if grid[row][col] == 0:
            shuffle(numberList)
            for value in numberList:
                if not (value in grid[row]):
                    if not value in [grid[r][col] for r in range(9)]:
                        square = [grid[r][col//3*3:col//3*3+3] for r in range(row//3*3, row//3*3+3)]
                        if not value in [num for sublist in square for num in sublist]:
                            grid[row][col] = value
                            if checkGrid(grid):
                                return True
                            else:
                                if fillGrid(grid):
                                    return True
            break
    grid[row][col] = 0

# Function to create a Sudoku puzzle
def create_puzzle(grid, attempts=5):
    while attempts > 0:
        row = randint(0, 8)
        col = randint(0, 8)
        while grid[row][col] == 0:
            row = randint(0, 8)
            col = randint(0, 8)
        backup = grid[row][col]
        grid[row][col] = 0

        copyGrid = [r.copy() for r in grid]
        counter = 0
        if not solveGrid(copyGrid):
            grid[row][col] = backup
            attempts -= 1
    return grid

# Generate and save multiple unique Sudoku puzzles and solutions
def generate_sudoku_puzzles(num_puzzles):
    puzzles_and_solutions = []
    for _ in range(num_puzzles):
        grid = create_empty_grid()
        fillGrid(grid)
        solution = [row.copy() for row in grid]
        puzzle = create_puzzle(grid)

        puzzles_and_solutions.append({"puzzle": puzzle, "solution": solution})
    
    with open('sudoku_puzzles.json', 'w') as json_file:
        json.dump(puzzles_and_solutions, json_file)

    print(f"{num_puzzles} Sudoku puzzles saved to sudoku_puzzles.json")

# Number of Sudoku puzzles to generate
num_puzzles = 3
generate_sudoku_puzzles(num_puzzles)