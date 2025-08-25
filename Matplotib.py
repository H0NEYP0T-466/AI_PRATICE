import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 4, 9, 16, 25]
y3 = [1, 8, 27, 64, 125]

plt.plot(x, y1, label="y = 2x")     # First line
plt.plot(x, y2, label="y = x^2")    # Second line
plt.plot(x, y3, label="y = x^3")    # Third line

plt.title("Legend Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.legend()   
plt.show()
