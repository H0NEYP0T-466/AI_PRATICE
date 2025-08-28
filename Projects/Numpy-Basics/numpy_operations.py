"""
NumPy Basics: Fundamental Array Operations and Mathematical Computations
=======================================================================

This project demonstrates core NumPy functionality including:
- Array creation and manipulation
- Mathematical operations
- Linear algebra computations
- Statistical calculations
- Random number generation
"""

import numpy as np
import os

def create_arrays_demo():
    """Demonstrate various ways to create NumPy arrays"""
    print("=" * 50)
    print("ARRAY CREATION DEMONSTRATIONS")
    print("=" * 50)
    
    # Basic array creation
    arr_1d = np.array([1, 2, 3, 4, 5])
    print(f"1D Array: {arr_1d}")
    
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"2D Array:\n{arr_2d}")
    
    # Built-in array creation functions
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    identity = np.eye(4)
    
    print(f"Zeros array (3x3):\n{zeros}")
    print(f"Ones array (2x4):\n{ones}")
    print(f"Identity matrix (4x4):\n{identity}")
    
    # Range arrays
    arange_arr = np.arange(0, 10, 2)  # start, stop, step
    linspace_arr = np.linspace(0, 1, 5)  # start, stop, num_points
    
    print(f"Arange (0 to 10, step 2): {arange_arr}")
    print(f"Linspace (0 to 1, 5 points): {linspace_arr}")
    
    return arr_1d, arr_2d

def mathematical_operations():
    """Demonstrate mathematical operations with NumPy"""
    print("\n" + "=" * 50)
    print("MATHEMATICAL OPERATIONS")
    print("=" * 50)
    
    # Create sample arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    
    print(f"Array a: {a}")
    print(f"Array b: {b}")
    
    # Basic arithmetic
    print(f"Addition (a + b): {a + b}")
    print(f"Subtraction (a - b): {a - b}")
    print(f"Multiplication (a * b): {a * b}")
    print(f"Division (a / b): {a / b}")
    print(f"Power (a ** 2): {a ** 2}")
    
    # Element-wise functions
    print(f"Square root of a: {np.sqrt(a)}")
    print(f"Exponential of a: {np.exp(a)}")
    print(f"Natural log of b: {np.log(b)}")
    print(f"Sine of a: {np.sin(a)}")
    
    return a, b

def linear_algebra_demo():
    """Demonstrate linear algebra operations"""
    print("\n" + "=" * 50)
    print("LINEAR ALGEBRA OPERATIONS")
    print("=" * 50)
    
    # Create matrices
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    vector = np.array([1, 2])
    
    print(f"Matrix A:\n{matrix_a}")
    print(f"Matrix B:\n{matrix_b}")
    print(f"Vector: {vector}")
    
    # Matrix operations
    matrix_mult = np.dot(matrix_a, matrix_b)
    print(f"Matrix multiplication (A @ B):\n{matrix_mult}")
    
    # Matrix-vector multiplication
    matrix_vector = np.dot(matrix_a, vector)
    print(f"Matrix-vector multiplication (A @ vector): {matrix_vector}")
    
    # Determinant and inverse
    det_a = np.linalg.det(matrix_a)
    inv_a = np.linalg.inv(matrix_a)
    
    print(f"Determinant of A: {det_a:.2f}")
    print(f"Inverse of A:\n{inv_a}")
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(matrix_a)
    print(f"Eigenvalues of A: {eigenvals}")
    print(f"Eigenvectors of A:\n{eigenvecs}")
    
    return matrix_a, matrix_b

def statistical_operations():
    """Demonstrate statistical calculations"""
    print("\n" + "=" * 50)
    print("STATISTICAL OPERATIONS")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)  # mean=100, std=15, size=1000
    
    print(f"Sample data (first 10 values): {data[:10]}")
    print(f"Data shape: {data.shape}")
    
    # Basic statistics
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Standard deviation: {np.std(data):.2f}")
    print(f"Variance: {np.var(data):.2f}")
    print(f"Minimum: {np.min(data):.2f}")
    print(f"Maximum: {np.max(data):.2f}")
    
    # Percentiles
    percentiles = np.percentile(data, [25, 50, 75])
    print(f"25th, 50th, 75th percentiles: {percentiles}")
    
    # Correlation coefficient (with another dataset)
    data2 = data + np.random.normal(0, 5, 1000)
    correlation = np.corrcoef(data, data2)[0, 1]
    print(f"Correlation with modified data: {correlation:.3f}")
    
    return data

def array_manipulation():
    """Demonstrate array manipulation techniques"""
    print("\n" + "=" * 50)
    print("ARRAY MANIPULATION")
    print("=" * 50)
    
    # Create sample array
    arr = np.arange(12).reshape(3, 4)
    print(f"Original array (3x4):\n{arr}")
    
    # Reshaping
    reshaped = arr.reshape(2, 6)
    print(f"Reshaped to (2x6):\n{reshaped}")
    
    # Flattening
    flattened = arr.flatten()
    print(f"Flattened: {flattened}")
    
    # Transposing
    transposed = arr.T
    print(f"Transposed:\n{transposed}")
    
    # Slicing and indexing
    print(f"First row: {arr[0, :]}")
    print(f"Last column: {arr[:, -1]}")
    print(f"Sub-array (first 2 rows, first 2 cols):\n{arr[:2, :2]}")
    
    # Boolean indexing
    mask = arr > 5
    print(f"Elements > 5: {arr[mask]}")
    
    # Concatenation
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    
    horizontal = np.hstack([arr1, arr2])
    vertical = np.vstack([arr1, arr2])
    
    print(f"Horizontal concatenation: {horizontal}")
    print(f"Vertical concatenation:\n{vertical}")
    
    return arr

def random_number_operations():
    """Demonstrate random number generation"""
    print("\n" + "=" * 50)
    print("RANDOM NUMBER OPERATIONS")
    print("=" * 50)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Various random distributions
    uniform = np.random.uniform(0, 1, 10)
    normal = np.random.normal(0, 1, 10)
    integers = np.random.randint(1, 100, 10)
    
    print(f"Uniform distribution [0,1]: {uniform}")
    print(f"Normal distribution (μ=0, σ=1): {normal}")
    print(f"Random integers [1,100): {integers}")
    
    # Random sampling
    population = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sample = np.random.choice(population, size=5, replace=False)
    print(f"Random sample without replacement: {sample}")
    
    # Random matrix
    random_matrix = np.random.rand(3, 3)
    print(f"Random 3x3 matrix:\n{random_matrix}")
    
    return uniform, normal, integers

def main():
    """Main function to run all demonstrations"""
    print("NumPy Basics: Comprehensive Demonstration")
    print("========================================")
    
    # Run all demonstrations
    arr_1d, arr_2d = create_arrays_demo()
    a, b = mathematical_operations()
    matrix_a, matrix_b = linear_algebra_demo()
    data = statistical_operations()
    arr = array_manipulation()
    uniform, normal, integers = random_number_operations()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("✅ Array creation and basic operations")
    print("✅ Mathematical computations")
    print("✅ Linear algebra operations")
    print("✅ Statistical calculations")
    print("✅ Array manipulation techniques")
    print("✅ Random number generation")
    print("\nNumPy basics demonstration completed successfully!")

if __name__ == "__main__":
    main()