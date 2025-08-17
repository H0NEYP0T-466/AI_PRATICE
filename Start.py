import numpy as np

arr1=np.array([12,12,12,12])
arr2=np.array([12,12,12,12])
print(arr1.size)
print(arr1+arr2)
print(type(arr1))


zero=np.zeros(5)
print(zero)

one=np.ones((2,2))
print(one)

arr_range = np.arange(101)
print (arr_range)


arr_step = np.arange(2, 21, 2)
print (arr_step)

lin2 = np.linspace(10, 50, 10)
print (lin2)

print("hello world")



arr = np.array([10, 20, 30, 40, 50])

print(arr[2])       # Single element → 30
print(arr[1:4])     # Slice from index 1 to 3 → [20 30 40]
print(arr[:3])      # First 3 elements → [10 20 30]
print(arr[-2:])     # Last 2 elements → [40 50]



print("hello world")
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(arr2d[0, 1])      # Row 0, Col 1 → 2
print(arr2d[:, 1])      # All rows, Col 1 → [2 5 8]
print(arr2d[1, :])      # Row 1, all cols → [4 5 6]
print(arr2d[0:2, 0:2])  # Top-left 2x2 block → [[1 2]
                        #                         [4 5]]




print("hello world")
arr = np.array([5, 10, 15, 20, 25])

print(arr > 15)       # Boolean mask → [False False False  True  True]
print(arr[arr > 15])  # Filtered values → [20 25]

