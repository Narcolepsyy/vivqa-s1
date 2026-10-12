import numpy as np
import os
import json
path = "./features/vinvl_vinvl/30.npy"
features = np.load(path, allow_pickle=True)[()]
print(features)
# Kiểm tra kiểu key của từng phần tử
print(all(isinstance(key, str) for key in features))
print(type(features))
print(features.keys())
# features = {str(key): value for key, value in features.items()}
# print(all(isinstance(key, str) for key in features))