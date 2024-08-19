from random import randint, shuffle
import json

# Initialize empty 9 by 9 grid
grid = [[0 for _ in range(9)] for _ in range(9)]

# A function to check if the grid is full
def checkGrid(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return False
    return True

# A backtracking/recursive function to check all possible combinations of numbers until a solution is found
def solveGrid(grid):
    global counter
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
                                counter += 1
                                break
                            else:
                                if solveGrid(grid):
                                    return True
            break
    grid[row][col] = 0

numberList = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# A backtracking/recursive function to fill the grid
def fillGrid(grid):
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

# Generate a fully solved grid
fillGrid(grid)

# Start removing numbers one by one to create a Sudoku puzzle
attempts = 5
counter = 1
while attempts > 0:
    row = randint(0, 8)
    col = randint(0, 8)
    while grid[row][col] == 0:
        row = randint(0, 8)
        col = randint(0, 8)
    backup = grid[row][col]
    grid[row][col] = 0

    copyGrid = [row.copy() for row in grid]
    counter = 0
    solveGrid(copyGrid)
    if counter != 1:
        grid[row][col] = backup
        attempts -= 1

print("Sudoku Grid Ready")

# Save the final grid to a JSON file
with open('sudoku_grid.json', 'w') as json_file:
    json.dump(grid, json_file)

print("Grid saved to sudoku_grid.json")