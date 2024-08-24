from datasets import load_dataset

ds = load_dataset("Ritvik19/Sudoku-Dataset", split='train')


for puzzle in ds['puzzle']:
    print(puzzle)