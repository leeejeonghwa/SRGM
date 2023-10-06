import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Define the modified Pham-Zhang IFD model function
def modified_pham_zhang_ifd_model(t, a, b, d):
    return a - a * np.exp(-b * t) * (1 + (b + d) * t + b * d * t**2)

# Load your observed cumulative failure data
# Replace this with your actual data
cumulative_failures_data = np.array([0, 80, 109, 135, 195, 210, 221, 240, 256, 264, 286, 296, 307, 324])

# Generate time values (timesteps)
timesteps = np.arange(len(cumulative_failures_data))

# Define the function for curve fitting
def fit_modified_pham_zhang_ifd(t, a, b, d):
    return modified_pham_zhang_ifd_model(t, a, b, d)

# Perform curve fitting to optimize parameters (a, b, d)
popt, pcov = curve_fit(fit_modified_pham_zhang_ifd, timesteps, cumulative_failures_data)

# Extract optimized parameters
a_optimized, b_optimized, d_optimized = popt

# Generate predictions using the optimized parameters
failures_predictions = modified_pham_zhang_ifd_model(timesteps, a_optimized, b_optimized, d_optimized)

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
print(f"Optimized Parameters - a: {a_optimized}, b: {b_optimized}, d: {d_optimized}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Bias: {bias}")
