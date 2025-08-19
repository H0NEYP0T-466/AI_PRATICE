import numpy as np

board = np.array([
    [5,3,0, 0,7,0, 0,0,0],
    [6,0,0, 1,9,5, 0,0,0],
    [0,9,8, 0,0,0, 0,6,0],
    
    [8,0,0, 0,6,0, 0,0,3],
    [4,0,0, 8,0,3, 0,0,1],
    [7,0,0, 0,2,0, 0,0,6],
    
    [0,6,0, 0,0,0, 2,8,0],
    [0,0,0, 4,1,9, 0,0,5],
    [0,0,0, 0,8,0, 0,7,9]
])

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return i, j
    return None

def is_valid(board, row, col, num):
    if num in board[row]:
        return False

    if num in board[:, col]:
        return False

    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False

    return True

def solve(board):
    empty = find_empty(board)
    if not empty:
        return True  
    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num

            if solve(board):  
                return True
            board[row, col] = 0  

    return False


print("Original Board:")
print(board)

if solve(board):
    print("\nSolved Board:")
    print(board)
else:
    print("No solution exists")

