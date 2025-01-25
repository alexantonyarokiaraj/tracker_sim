import matplotlib.pyplot as plt

# X values (common for all)
x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Y values
a = [8.691, 7.694, 8.633, 8.666, 8.671, 8.671, 7.671, 8.671, 8.671, 8.671]
a1 = [7.921, 3.663, 0.8821, 0.7609, 0.7644, 0.7644, 0.7644, 0.7644, 0.7644, 0.7644]
b = [9.047, 5.199, 5.01, 6.119, 6.844, 6.876, 6.876, 6.876, 6.876, 6.876]
b1 = [8.31, 4.509, 1.771, 0.707, 0.4263, 0.4326, 0.4328, 0.4328, 0.4328, 0.4328]
c = [9.519, 4.473, 3.187, 3.206, 3.537, 4.259, 4.878, 4.921, 4.921, 4.921]
c1 = [6.981, 3.793, 1.717, 0.9202, 0.3801, 0.01064, -0.07642, -0.064, -0.06386, -0.06386]
d = [9.676, 4.085, 2.51, 2.17, 2.201, 2.368, 2.599, 2.89, 3.382, 3.604]
d1 = [6.303, 3.37, 1.543, 0.8054, 0.453, 0.257, 0.1142, 0.017718, -0.01047, -0.1122]

# Plot the first set (a, b, c, d)
plt.figure(figsize=(10, 5))
plt.plot(x, a, marker='o', linestyle='-', label='a')
plt.plot(x, b, marker='s', linestyle='-', label='b')
plt.plot(x, c, marker='^', linestyle='-', label='c')
plt.plot(x, d, marker='d', linestyle='-', label='d')

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Plot of a, b, c, d as a function of X")
plt.legend()
plt.grid(True)
plt.show()

# Plot the second set (a1, b1, c1, d1)
plt.figure(figsize=(10, 5))
plt.plot(x, a1, marker='o', linestyle='-', label='a1')
plt.plot(x, b1, marker='s', linestyle='-', label='b1')
plt.plot(x, c1, marker='^', linestyle='-', label='c1')
plt.plot(x, d1, marker='d', linestyle='-', label='d1')

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Plot of a1, b1, c1, d1 as a function of X")
plt.legend()
plt.grid(True)
plt.show()
