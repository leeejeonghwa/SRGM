# Aij 값을 NumPy 배열로 변환
# 전체 모델 한번에 사용 및 데이터 저장



import numpy as np

from M1 import M1_results_list
from M10 import M10_results_list
from M11 import M11_results_list
from M12 import M12_results_list
from M13 import M13_results_list
from M14 import M14_results_list
from M15 import M15_results_list
from M2 import M2_results_list
from M3 import M3_results_list
from M4 import M4_results_list
from M5 import M5_results_list
from M6 import M6_results_list
from M7 import M7_results_list
from M8 import M8_results_list
from M9 import M9_results_list




Aij = np.abs(np.array([M1_results_list, M2_results_list, M3_results_list, M4_results_list, M5_results_list, M6_results_list, M7_results_list, M8_results_list, M9_results_list, M10_results_list, M11_results_list, M12_results_list, M13_results_list, M14_results_list, M15_results_list]))

np.save("Aij.npy", Aij)