import numpy as np

# m, n, 및 Aij 값 설정
m = 15  # 대안 개수
n = 9  # 성능 지표 개수


def test(th):
    Aij_values = [
        [69.41, 11047.23, 21.30, 0.53, 2.35, 2.28, 163.13, 0.19, 33.83],
        [10.11, 493.24, 34.43, 0.12, 3.62, 0.40, 28.93, 0.95, 7.16],
        [30.02, 10220.44, 120.07, 0.37, 10 * 10 ** 12, 3.49, 116.14, 0.10, 32.54],
        [30.02, 10220.44, 120.07, 0.37, 26 * 10 ** 15, 3.49, 116.14, 0.10, 32.54],
        [30.02, 10220.44, 120.07, 0.37, 78 * 10 ** 14, 3.49, 116.14, 0.10, 32.54],
        [222.01, 33053322.50, 298.60, 0.89, 3.70, 554.15, 5885.98, 3544.59, 1850.31],
        [3.76, 101.65, 14.32, 0.05, 2.40, 0.07, 12.34, 0.99, 3.28],
        [16.37, 2040.01, 62.17, 0.27, 1.19, 0.64, 54.50, 0.78, 14.55],
        [30.02, 10220.44, 120.07, 0.37, 97933.71, 3.49, 116.14, 0.10, 32.54],
        [18.76, 1578.68, 57.10, 0.20, 7.55, 9.23, 52.46, 0.83, 12.80],
        [0.46, 37.30, 2.43, 0.01, 1.51, 0.08, 6.53, 1.00, 2.04],
        [30.02, 10220.44, 120.07, 0.37, 22 * 10 ** 16, 3.49, 116.14, 0.10, 32.54],
        [3.49, 4558.43, 88.15, 0.32, 0.10, 4.45, 69.27, 0.51, 21.74],
        [2.58, 128.20, 16.33, 0.07, 2.87, 0.05, 12.64, 0.99, 3.71],
        [30.02, 10220.44, 120.07, 0.37, 68 * 10 ** 15, 3.49, 116.14, 0.10, 32.54]

        # [295.76,113488.52,336.63,0.84,2.17,16.20,629.92,2356.33,2360.19],
        #  [0.09,1.59,0.76,0.07,0.08,0.13,0.68,0.99,4.50],
        #  [0.05,2.65,0.60,0.06,1.22,0.34,0.62,0.99,4.15],
        #  [2.47,51.26,9.40,0.55,14*10**15,10.31,8.76,0.13,51.61],
        #  [2.47,51.26,9.40,0.55,30*10**13,10.31,8.76,0.13,51.61],
        #  [25.29,824.90,35.37,0.17,0.00,8.69,53.84,16.20,201.58],
        #  [0.07,1.58,0.75,0.07,0.10,0.14,0.67,0.99,4.52],
        #  [0.54,0.93,1.98,0.23,0.60,0.15,1.76,0.96,10.09],
        #  [2.47,51.26,9.40,0.55,12*10**5,10.31,8.76,0.13,51.61],
        #  [0.47,1.03,1.10,0.03,1.92,4.88,1.31,0.98,6.90],
        #  [0.19,2.38,0.91,0.13,2.66,0.08,0.88,0.99,5.50],
        #  [2.47,51.26,9.40,0.62,98.84,10.31,8.76,0.13,51.61],
        #  [0.17,3.45,0.87,0.12,0.84,0.07,0.82,0.99,5.20],
        #  [1.83,3.08,0.97,0.04,7.00,74.45,4.37,0.83,19.92],
        #  [0.15,2.57,0.81,0.09,0.00,0.12,0.72,0.99,4.60],
        #  [124.51,29504.49,256.19,0.18,0.38,14.05,283.27,611.94,1203.50]
    ]

    np.set_printoptions(formatter={'all': lambda x: f'{x:.20f}'})

    # Aij 값을 numpy 배열로 변환
    Aij = np.array(Aij_values)
    # print(f"기본 배열 {Aij}")
    # w_i = np.array([0.0836, 0.2488, 0.2164, 0.0026, 0.0941,0.0795, 0.0121, 0.2306, 0.0323 ])
    # w_i = np.array([0.0049,0.0198,0.0123,0.0021,0.259,0.0128,0.8974,0.0130,0.0118])
    w_i = np.array(
        [0.207203006, 0.002330204, 0.245622032, 0.253896042, 0.090787833, 0.039327839, 0.083650056, 0.001889527,
         0.075293462])

    norm_Aij = np.zeros_like(Aij)
    for j in range(n):
        if j in [0, 1, 2, 3, 4, 5, 6, 8]:  #
            norm_Aij[:, j] = Aij[:, j] / np.max(Aij[:, j])
            # print(f"한열의 최대값 {np.max(Aij[:, j])}")
        else:
            norm_Aij[:, j] = np.min(Aij[:, j]) / Aij[:, j]
            # print(f"한열의 최소 값 {np.min(Aij[:, j])}")

    # print(f"정규화 행렬{norm_Aij}")

    # 가중합 계산
    weighted_Aij = w_i * norm_Aij
    # print(f"가중치 행렬{weighted_Aij}")

    # 각 열마다 최소값 계산하여 nsj에 저장
    nsj = np.min(weighted_Aij, axis=0)
    # print("최솟값")
    # print(nsj)

    # Initialize EDi and TDi arrays
    ED_i = np.zeros(m)
    TD_i = np.zeros(m)

    # Calculate EDi and TDi
    for i in range(m):
        ED_i[i] = np.sqrt(np.sum((weighted_Aij[i, :] - nsj) ** 2))
        TD_i[i] = np.sum(np.abs(weighted_Aij[i, :] - nsj))

    # print(f"유클리드{ED_i}")
    # print(f"맨해튼{TD_i}")

    # Initialize Rmxn and ymxm matrices
    R = np.zeros((m, m))
    y = np.zeros((m, m))

    # Calculate Rmxn and ymxm
    # th = 0.035
    for i in range(m):
        for k in range(m):
            y[i, k] = ED_i[i] - ED_i[k]
            # if abs(y[i, k]) < th:
            # print(f"i: {i} k: {k} -> {y[i, k]}")
            if abs(y[i, k]) >= th:
                phi_y = 1
            else:
                phi_y = 0
            R[i, k] = (ED_i[i] - ED_i[k]) + (phi_y * (TD_i[i] - TD_i[k]))

    # Initialize Pmx1
    P = np.zeros(m)

    # print(R)

    # Calculate Pmx1
    for i in range(m):
        P[i] = np.sum(R[i, :])

    # print("P------------------------")
    # print(P)

    result = list(enumerate(list(P), start=1))
    sorted_result = sorted(result, key=lambda x: x[1])
    data = {}
    for idx, val in enumerate(sorted_result):
        print(idx, val)
        data[idx] = val[0]
    return data

# Output: Ranking of alternatives
# ranked_indices = np.argsort(P[:, 0])  # Sort in ascending order
# ranked_alternatives = [i + 1 for i in ranked_indices]  # Adding 1 to match 1-based indexing

# # Print the ranked alternatives in ascending order
# print("Ranked Alternatives (Ascending Order):")
# for rank, alternative in enumerate(ranked_alternatives, start=1):
#     print(f"Rank {rank}: Alternative {alternative}, {P[alternative-1]}")
