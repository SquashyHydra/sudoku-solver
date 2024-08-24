import json
from datasets import load_dataset

name = "sudoku_data_validation"

def format_sudoku(sudoku_string):
    return [[int(sudoku_string[i * 9 + j]) for j in range(9)] for i in range(9)]

print(f'Loading Dataset')
ds = load_dataset("Ritvik19/Sudoku-Dataset")
print(f'Loading Puzzles and Solutions')
puzzle_train = ds['train']['puzzle']
solutions_train = ds['train']['solution']

puzzle_validation = ds['validation']['puzzle']
solutions_validation = ds['validation']['solution']

formatted_data = []

print(f'Creating New Dataset')
if 'train' in name:
    for puzzle, solution in zip(puzzle_train, solutions_train):
        formatted_puzzle = format_sudoku(puzzle)
        formatted_solution = format_sudoku(solution)
        
        formatted_data.append({
            'puzzle': formatted_puzzle,
            'solution': formatted_solution
        })
elif 'validation' in name:
    for puzzle, solution in zip(puzzle_validation, solutions_validation):
        formatted_puzzle = format_sudoku(puzzle)
        formatted_solution = format_sudoku(solution)
        
        formatted_data.append({
            'puzzle': formatted_puzzle,
            'solution': formatted_solution
        })

print(f"Saving Dataset")
with open(f'{name}.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)

print(f"Data has been successfully formatted and saved to {name}.json")