import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Define the Pham-Zhang model function for cumulative data
def pham_zhang_model_cumulative(t, a, b, c, alpha, beta):
    numerator = (c + a) * (1 - np.exp(-b * t)) - a * b / (b - alpha) * (np.exp(-alpha * t) - np.exp(-b * t))
    denominator = (1 + beta * np.exp(-b * t))
    return numerator / denominator

# Load your observed cumulative failure data
# Replace this with your actual data
cumulative_failures_data = np.array([ 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 22, 22])

# Generate time values (timesteps)
timesteps = np.arange(len(cumulative_failures_data))

# Perform curve fitting to optimize parameters (a, b, c, alpha, beta)
# Use initial guess values for parameters
initial_guess = [1, 1, 1, 1, -1]  # You may need to adjust these initial values
popt, pcov = curve_fit(pham_zhang_model_cumulative, timesteps, cumulative_failures_data, p0=initial_guess, bounds=([0, 0, 0, 0, -np.inf], [np.inf, 1, np.inf, np.inf, 0]))

# Extract optimized parameters
a_optimized, b_optimized, c_optimized, alpha_optimized, beta_optimized = popt

# Generate predictions using the optimized parameters
failures_predictions = pham_zhang_model_cumulative(timesteps, a_optimized, b_optimized, c_optimized, alpha_optimized, beta_optimized)

# Calculate a performance metric (e.g., Mean Squared Error)
mse = mean_squared_error(cumulative_failures_data, failures_predictions)
bias = np.mean(np.abs(failures_predictions - cumulative_failures_data))

# Plot observed cumulative failures and predictions
plt.scatter(timesteps, cumulative_failures_data, label='Observed Cumulative Failures')
plt.plot(timesteps, failures_predictions, label='Predicted Cumulative Failures (Optimized Parameters)', color='red')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Failures')
plt.legend()
plt.show()

# Display the performance metric (MSE in this case)
print(f"Optimized Parameters - a: {a_optimized}, b: {b_optimized}, c: {c_optimized}, alpha: {alpha_optimized}, beta: {beta_optimized}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Bias: {bias}")
