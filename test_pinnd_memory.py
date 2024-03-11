import ctypes

import numpy as np
from cuda import cudart

data = np.zeros([2, 3], dtype=np.float32)
nElement = np.prod(data.shape)
nByteSize = np.nbytes[np.float32] * nElement

# 申请页锁定内存（Pinned memory）
_, pBuffer = cudart.cudaHostAlloc(nByteSize, cudart.cudaHostAllocWriteCombined)
print(pBuffer)
