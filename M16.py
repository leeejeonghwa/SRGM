import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Define the Zhang, Teng & Pham (2003) model function for cumulative data
def zhang_teng_pham_2003_model_cumulative(t, a, p, beta, alpha, b, c):
    numerator = 1 - ((1 + alpha) * np.exp(-b * t)) / ((1 + alpha * np.exp(-b * t))**c / (b * (p - beta)))
    return a / (p - beta) * numerator

# Load your observed cumulative failure data
# Replace this with your actual data
cumulative_failures_data = np.array([1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 22, 22])

# Generate time values (timesteps)
timesteps = np.arange(len(cumulative_failures_data))

# Define fixed parameters for the Zhang, Teng & Pham (2003) model
a_fixed = 0.510 # Replace with your desired value
p_fixed = 22  # Replace with your desired value
beta_fixed = 0.996 # Replace with your desired value
alpha_fixed = -0.606  # Replace with your desired value
b_fixed = 0.116  # Replace with your desired value
c_fixed =0.790  # Replace with your desired value

# Generate predictions using fixed parameters and time values
failures_predictions = zhang_teng_pham_2003_model_cumulative(timesteps, a_fixed, p_fixed, beta_fixed, alpha_fixed, b_fixed, c_fixed)

# Calculate a performance metric (e.g., Mean Squared Error)
mse = mean_squared_error(cumulative_failures_data, failures_predictions)
bias = np.mean(np.abs(failures_predictions - cumulative_failures_data))

# Plot observed cumulative failures and predictions
plt.scatter(timesteps, cumulative_failures_data, label='Observed Cumulative Failures')
plt.plot(timesteps, failures_predictions, label='Predicted Cumulative Failures (Fixed Parameters)', color='red')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Failures')
plt.legend()
plt.show()

# Display the performance metric (MSE in this case)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Bias: {bias}")

