import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Define the Zhang, Teng & Pham (2003) model function for cumulative data
def zhang_teng_pham_model_cumulative(t, a, beta, alpha, b, c):
    p=2 #p값은 2,3,4만 가능
    numerator = 1 - ((1 + alpha) * np.exp(-b * t)) / ((1 + alpha * np.exp(-b * t))**c / (b * (p - beta)))
    return a / (p - beta) * numerator

cumulative_failures_data = np.array([0, 80, 109, 135, 195, 210, 221, 240, 256, 264, 286, 296, 307, 324, 328, 343, 346, 356, 370, 375, 388, 402, 412, 421])

# Generate time values (timesteps)
timesteps = np.arange(len(cumulative_failures_data))

# Specify the index where you want to split the data into training and testing
split_index = 20# For example, split after the 15th data point

X, y= timesteps, cumulative_failures_data
# Split the data into training and testing sets
X_train, y_train = timesteps[:split_index], cumulative_failures_data[:split_index]
X_test, y_test = timesteps[split_index:], cumulative_failures_data[split_index:]


# Perform curve fitting to optimize parameters (a and b) using only the training data
popt, pcov = curve_fit(zhang_teng_pham_model_cumulative, X_train, y_train,method = 'trf', bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]), max_nfev=10000)

# Extract optimized parameters
a_optimized, beta_optimized, alpha_optimized, b_optimized, c_optimized= popt

# Generate predictions using the optimized parameters for both training and testing data
failures_predictions_train = zhang_teng_pham_model_cumulative(X_train, a_optimized, beta_optimized, alpha_optimized, b_optimized, c_optimized)
failures_predictions_test = zhang_teng_pham_model_cumulative(X_test, a_optimized, beta_optimized, alpha_optimized, b_optimized, c_optimized)

k = len(X_test) # 데이터 포인트 수
p_data = failures_predictions_test  # 모델의 예측값
r_data= y_test  # 실제 값
p= 2 #파라미터의 갯수

bias = np.sum(p_data - r_data) / k

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(r_data, p_data)

# Calculate Mean Error of Prediction (MEOP)
meop = np.sum(np.abs(p_data-r_data)) / (k - p + 1)

# Calculate Absolute Error (AE)
ae = np.mean(np.abs(p_data - r_data))

# Calculate Percent Relative Error (PRR)
prr = np.sum((p_data - r_data) / p_data)

# Calculate Variance
variance_numerator = np.sum((failures_predictions_test - y_test - bias)**2)
variance_denominator = len(X_test) - 1
variance = np.sqrt(variance_numerator / variance_denominator)


# Calculate R-squared (Rsq)
numerator = np.sum((y_test - failures_predictions_test)**2)
denominator = np.sum((y_test - np.mean(y_test))**2)
rsq = 1 - (numerator / denominator)

# Calculate True Skill Statistic (TS)
numerator = np.sum((failures_predictions_test - y_test)**2)
denominator = np.sum(y_test**2)
ts = np.sqrt(numerator / denominator) * 100

# a_optimized, beta_optimized, alpha_optimized, b_optimized, c_optimized
# Calculate Noise (Standard Deviation of Residuals)
noise = 0
for i in range(1, len(X_test)):
    lambda_ti = a_optimized*c_optimized/(1+alpha_optimized*np.exp(b_optimized*i))*((((1+alpha_optimized)*np.exp(-1 * b_optimized*i))/(1+alpha_optimized*np.exp(-b_optimized*i)))**((c_optimized/b_optimized)*(2-beta_optimized))) # 현재 시간 스텝에서 모델의 예측값

    lambda_ti_minus_1 = a_optimized*c_optimized/(1+alpha_optimized*np.exp(b_optimized*(i-1)))*((((1+alpha_optimized)*np.exp(-1 * b_optimized*i))/(1+alpha_optimized*np.exp(-b_optimized*i)))**((c_optimized/b_optimized)*(2-beta_optimized))) # 이전 시간 스텝에서 모델의 예측값

    if lambda_ti_minus_1 != 0:
        noise += np.abs((lambda_ti - lambda_ti_minus_1) / lambda_ti_minus_1)

# Plot observed cumulative failures and predictions for testing data
plt.scatter(X, y, label='Observed Cumulative Failures (Training Data)')
plt.plot(X_train, failures_predictions_train, label='Predicted Cumulative Failures (Optimized Parameters)', color='red')
plt.plot(X_test, failures_predictions_test, label='Predicted Cumulative Failures (Optimized Parameters)', color='green')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Failures')
plt.legend()
plt.show()

# Display the performance metrics
print(f"Optimized Parameters - a: {a_optimized}, b: {beta_optimized}, c: {alpha_optimized}, d:{b_optimized}, e:{c_optimized}")
print("--------------------------------------------------------------------------------")
print("Bias:", round(bias, 3))
print(f"Mean Squared Error (MSE): {round(mse, 3)}")
print(f"Mean Error of Prediction (MEOP): {round(meop, 3)}")
print(f"Absolute Error (AE): {round(ae, 3)}")
print(f"Noise (Standard Deviation of Residuals): {round(noise, 3)}")
print(f"Percent Relative Error (PRR): {round(prr, 3)}")
print(f"Variance: {round(variance, 3)}")
print(f"R-squared (Rsq): {round(rsq, 3)}")
print(f"True Skill Statistic (TS): {round(ts, 3)}")


M16_results_list = [round(bias, 3), round(mse, 3),round(meop, 3),round(ae, 3),round(noise, 3),round(prr, 3),round(variance, 3),round(rsq, 3),round(ts, 3)]