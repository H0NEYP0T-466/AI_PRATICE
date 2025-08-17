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

arr = np.array([100, 200, 300, 400, 500])

print(arr[[0, 2, 4]])   # Pick elements at indices 0,2,4 → [100 300 500]

arr2d = np.array([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

print(arr2d[[0, 2], [1, 2]])  # Picks → [20 (row0,col1), 90 (row2,col2)]

# a.shape   # (5,)
# a.ndim    # 1
# a.size    # 5
# a.dtype   # e.g., dtype('int64')

# b.shape   # (3, 4)
# b.ndim    # 2
# b.size    # 12
# b.dtype   # e.g., dtype('int64')



import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)   # [11 22 33 44]
print(a - b)   # [-9 -18 -27 -36]
print(a * b)   # [10 40 90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]
print(a ** 2)  # [ 1  4  9 16]



arr = np.array([1, 2, 3])

print(arr + 10)        # Scalar broadcast → [11 12 13]

mat = np.array([[1, 2, 3],
                [4, 5, 6]])

print(mat + arr)       # Broadcast row → [[2 4 6]
                       #                   [5 7 9]]

print(mat * 2)         # Multiply each element → [[2 4 6]
                       #                          [8 10 12]]

arr = np.array([1, 2, 3, 4, 5])

print(np.sum(arr))   # 15
print(np.mean(arr))  # 3.0
print(np.std(arr))   # 1.414...
print(np.min(arr))   # 1
print(np.max(arr))   # 5


data = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(np.sum(data))          # Total sum → 21
print(np.sum(data, axis=0))  # Column-wise → [5 7 9]
print(np.sum(data, axis=1))  # Row-wise → [6 15]


import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32
print(a @ b)          # Same as dot


A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(np.matmul(A, B))
# [[19 22]
#  [43 50]]

M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(M.T)
# [[1 4]
#  [2 5]
#  [3 6]]

A = np.array([[1, 2],
              [3, 4]])

print(np.linalg.det(A))   # Determinant = -2.0
print(np.linalg.inv(A))   # Inverse
# [[-2.   1. ]
#  [ 1.5 -0.5]]

v = np.array([3, 4])
print(np.linalg.norm(v))  # √(3²+4²) = 5


A = np.array([[4, -2],
              [1,  1]])

values, vectors = np.linalg.eig(A)

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)


print("Random numbers:")

arr = np.random.rand(5,2)  
print(arr)

print("normal distrubution:")

arr = np.random.randn(10)  # 1000 random numbers (mean=0, std=1)

print(arr)
print(np.mean(arr))  # ≈ 0 (close to mean)
print(np.std(arr))   # ≈ 1 (close to std)


print("Random integers:")
np.random.seed(4)
arr = np.random.randint(1, 10, size=5)  # 5 numbers between 1–9
print(arr)   # e.g., [3 7 1 9 4]


print("Random seed:")
np.random.seed(42)
print(np.random.randn(3))  # Always the same numbers



print("Random choice:")
nums = np.array([10, 20, 30, 40, 50])

print(np.random.choice(nums))            # One random pick
print(np.random.choice(nums, 3))         # Pick 3 random values
print(np.random.choice(nums, 3, replace=False))  # 3 unique picks



print("seeding")
np.random.seed(42)
print(np.random.rand(3))  # Always the same numbers

print("seeding loop")

np.random.seed(0)
print(np.random.randint(1, 10, size=5))
print("hel")
for i in range(5):
    np.random.seed(i)
    print(np.random.randint(1, 10, size=5))



print("splitingg")
arr = np.arange(9)

print(np.array_split(arr, 4))  
print(np.split(arr, 3))  


a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.vstack([a,b]))
# [[1 2 3]
#  [4 5 6]]

print(np.hstack([a,b]))
# [1 2 3 4 5 6]



arr = np.arange(6)   # [0 1 2 3 4 5]

print(arr.reshape(2,3))  
# [[0 1 2]
#  [3 4 5]]
mat = arr.reshape(2,3)
print(mat.ravel())   # [0 1 2 3 4 5]
print(mat.flatten())


arr = np.array([1,2,2,3,3,3,4])
print(np.unique(arr))  # [1 2 3 4]


arr = np.array([1,2,3,4,5])

np.save("my_array.npy", arr)  # Save
print("saved")
loaded = np.load("my_array.npy")  # Load
print("loaded")

print(loaded)  # [1 2 3 4 5]
