from board_creator import create_sudoku
from sudokuAI import SudokuAI
"""
Reset grid:
        grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0]] 
"""
  
# Driver Code 
if __name__ == '__main__':
    def ask_grid():
        grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 0]] 

        rand_grid = input('Use Random Grid (Yes | No): ')
        if rand_grid.lower() in ["yes", "y"]:
            grid = create_sudoku()
        return grid

    def run_once():
        grid = ask_grid()
        sudoku = SudokuAI(grid)
        print("\nInitial Grid:")
        sudoku.print_grid()
        if not sudoku.solve():
            print("No solution exists")
        else:
            print("Solved Grid:")
            sudoku.print_grid()

    def multi_run():
        from time import sleep
        while True:
            try:
                grid = create_sudoku()
                sudoku = SudokuAI(grid)
                print("\nInitial Grid:")
                sudoku.print_grid()
                if not sudoku.solve():
                    print("No solution exists")
                else:
                    print("Solved Grid:")
                    sudoku.print_grid()
                    sleep(1)
            except KeyboardInterrupt:
                break

    run_once()