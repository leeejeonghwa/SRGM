import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Define the Yamada Imperfect Debugging 2 model function for cumulative data
def yamada_imperfect_debugging2_model(t, a, b, c):
    return a * (1 - np.exp(-b * t)) * (1 - (a / b)) + c * a * t

# Load your observed cumulative failure data
# Replace this with your actual data
cumulative_failures_data = np.array([0,16,24,27,33,41,49,54,58,69,75,81,86,90,93,96,98,99,100,100,100])


# Generate time values (timesteps)
timesteps = np.arange(len(cumulative_failures_data))

# Specify the index where you want to split the data into training and testing
split_index = 12# For example, split after the 15th data point

X, y= timesteps, cumulative_failures_data
# Split the data into training and testing sets
X_train, y_train = timesteps[:split_index], cumulative_failures_data[:split_index]
X_test, y_test = timesteps, cumulative_failures_data


# Perform curve fitting to optimize parameters (a and b) using only the training data
popt, pcov = curve_fit(yamada_imperfect_debugging2_model, X_train, y_train,method = 'trf',bounds=(0 , np.inf))

# Extract optimized parameters
a_optimized, b_optimized ,c_optimized = popt

# Generate predictions using the optimized parameters for both training and testing data
failures_predictions_train = yamada_imperfect_debugging2_model(X_train, a_optimized, b_optimized,c_optimized )
failures_predictions_test = yamada_imperfect_debugging2_model(X_test, a_optimized, b_optimized,c_optimized )

k = len(X_test) # 데이터 포인트 수
p_data = failures_predictions_test  # 모델의 예측값
r_data= y_test  # 실제 값
p= 2  #파라미터의 갯수

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
print(f"Optimized Parameters - a: {a_optimized}, b: {b_optimized},c:{c_optimized}")

# Print the bias
print("Bias:", bias)
