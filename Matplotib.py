import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Basic line plot
plt.plot(x, y, label="y = 2x")

# Titles & labels
plt.title("Basic Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Legend
plt.legend()

# Save the figure
#plt.savefig("plot.png")

# Show plot
plt.show()


