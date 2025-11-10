import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1. 读取 .npy 文件
test = np.load(r"D:\FPSO\Dataset\SVM\coherence_samples\0_001.npy")

# 2. 转成 PyTorch 张量
test = torch.tensor(test, dtype=torch.float32).unsqueeze(0)  # [1, 513]
print("张量形状:", test.shape)

# 3. 生成时间序列横轴
t = np.arange(test.shape[1])

# 4. 绘制曲线
plt.figure(figsize=(10, 4))
plt.plot(t, test.squeeze(0).numpy(), linewidth=1.5)
plt.title("时序信号曲线 (0_001.npy)")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

