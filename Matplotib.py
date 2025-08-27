import matplotlib.pyplot as plt
import numpy as np

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
fig1=plt.gcf()  # Get the current figure
#plt.savefig("legend_example.png")  # Save the figure as a PNG file

plt.close("fig1")

epochs = [1, 2, 3, 4, 5]
train_loss = [0.9, 0.7, 0.5, 0.4, 0.3]
val_loss = [1.0, 0.8, 0.6, 0.55, 0.5]

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


fig2 = plt.gcf()  # Get the current figure
plt.close("fig2")
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, c="blue", marker="o", label="Data Points")
plt.xlabel("X Feature")
plt.ylabel("Y Feature")
plt.title("Scatter Plot Example")
plt.legend()
plt.show()


fig3=plt.gcf()  # Get the current figure
plt.close("fig3")
#data = np.random.randn(1000)  

data=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.hist(data, bins=10, color="purple", edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram Example")
plt.show()


fig4=plt.gcf()  # Get the current figure
plt.close("fig4")

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, 
         color="red",        # line color
         marker="o",         # circle marker
         linestyle="--",     # dashed line
         linewidth=2, 
         markersize=8,
         label="Styled Line")

plt.title("Customized Line Plot")
plt.legend()
plt.show()

fig5=plt.gcf()  # Get the current figure
plt.close("fig5")


plt.plot(x, y, marker="s", color="green")

plt.xlim(0, 6)              # x-axis range
plt.ylim(0, 12)             # y-axis range
plt.xticks([0, 2, 4, 6])    # custom ticks
plt.yticks([0, 5, 10])      

plt.title("Axis Customization Example")
plt.show()

fig6=plt.gcf()  # Get the current figure
plt.close("fig6")

# Method 1: subplot()
plt.subplot(1, 2, 1)   # 1 row, 2 cols, first plot
plt.plot(x, y, color="blue")
plt.title("Plot 1")

plt.subplot(1, 2, 2)   # second plot
plt.plot(y, x, color="orange")
plt.title("Plot 2")

plt.tight_layout()
plt.show()

# Method 2: subplots()
fig, ax = plt.subplots(2, 1, figsize=(6,6))  # 2 rows, 1 col
ax[0].plot(x, y, color="purple")
ax[0].set_title("First Plot")

ax[1].scatter(y, x, color="brown")
ax[1].set_title("Second Plot")

plt.tight_layout()
plt.show()


