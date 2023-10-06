import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Define the Pham Nordmann Zhang model function for cumulative data
def pham_nordmann_zhang_model(t, a, b, beta, alpha):
    numerator = a * (1 - np.exp(-b * t)) * (a - (a / b)) + (alpha * a * t)
    denominator = 1 + (beta * np.exp(-b * t))
    return numerator / denominator

# Load your observed cumulative failure data
cumulative_failures_data = np.array([0, 80, 109, 135, 195, 210, 221, 240, 256, 264, 286, 296, 307, 324, 328, 343, 346, 356, 370, 375, 388, 402, 412, 421])

# Generate time values (timesteps)
timesteps = np.arange(len(cumulative_failures_data))

# Specify the index where you want to split the data into training and testing
split_index = 20# For example, split after the 15th data point

X, y= timesteps, cumulative_failures_data
# Split the data into training and testing sets
X_train, y_train = timesteps[:split_index], cumulative_failures_data[:split_index]
X_test, y_test = timesteps[split_index-1:], cumulative_failures_data[split_index-1:]


# Perform curve fitting to optimize parameters (a and b) using only the training data
popt, pcov = curve_fit(pham_nordmann_zhang_model, X_train, y_train,method = 'trf', bounds=([0, 0, 0, 0], [np.inf, 1, 1, np.inf]))

# Extract optimized parameters
a_optimized, b_optimized, c_optimized, d_optimized = popt

# Generate predictions using the optimized parameters for both training and testing data
failures_predictions_train = pham_nordmann_zhang_model(X_train, a_optimized, b_optimized, c_optimized, d_optimized)
failures_predictions_test = pham_nordmann_zhang_model(X_test, a_optimized, b_optimized, c_optimized, d_optimized)

k = len(X_test) # 데이터 포인트 수
p_data = failures_predictions_test  # 모델의 예측값
r_data= y_test  # 실제 값
p= 4  #파라미터의 갯수

bias = np.sum(p_data - r_data)/k


# Plot observed cumulative failures and predictions for testing data
plt.scatter(X, y, label='Observed Cumulative Failures (Traning Data)')
plt.plot(X_train, failures_predictions_train, label='Predicted Cumulative Failures (Optimized Parameters)', color='red')
plt.plot(X_test,failures_predictions_test , label='Predicted Cumulative Failures (Optimized Parameters)', color='green')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Failures')
plt.legend()
plt.show()

# Display the performance metrics
print(f"Optimized Parameters - a: {a_optimized}, b: {b_optimized}, c:{c_optimized}, d:{d_optimized}")

# Print the bias
print("Bias:", bias)