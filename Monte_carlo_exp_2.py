import numpy as np
import matplotlib.pyplot as plt

# Define the problem: Estimating the value of Pi
def estimate_pi(num_simulations):
    inside_circle = 0

    for _ in range(num_simulations):
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 <= 1:
            inside_circle += 1

    pi_estimate = (inside_circle / num_simulations) * 4
    return pi_estimate

# Run the simulation
num_simulations = 10000
pi_estimate = estimate_pi(num_simulations)

print(f"Estimated value of Pi after {num_simulations} simulations: {pi_estimate}")

# Visualization
x = np.random.uniform(-1, 1, num_simulations)
y = np.random.uniform(-1, 1, num_simulations)
inside_circle = x**2 + y**2 <= 1

plt.figure(figsize=(8, 8))
plt.scatter(x[inside_circle], y[inside_circle], color='blue', s=1)
plt.scatter(x[~inside_circle], y[~inside_circle], color='red', s=1)
plt.title(f"Monte Carlo Simulation: Estimation of Pi\nEstimated Pi = {pi_estimate}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

