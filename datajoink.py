import json
from datasets import load_dataset

ds = load_dataset("Ritvik19/Sudoku-Dataset", split='train')

def format_sudoku(sudoku_string):
    return [[int(sudoku_string[i * 9 + j]) for j in range(9)] for i in range(9)]

for puzzle in ds['puzzle']:
    print(format_sudoku(puzzle))