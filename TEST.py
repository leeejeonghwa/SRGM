import math

import numpy as np

from main import test
import json


# th = 0.02
# result = {}
# while th <= 0.05:
#     data = test(th)
#     result[th] = data
#     th += 0.00001
#
# print(result)
#
# with open("result.json", 'w') as json_file:
#     json.dump(result, json_file, indent=4)


test(0.045)


# print(math.log(np.array([[2.717, 2.171, 2.171]])))