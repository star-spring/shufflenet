import os
import re

# 根目录路径（改成你的实际路径）
root_folder = r"E:\BaiduNetdisk\Fault-diagnosis-based-on-deep-learning-main ResDSC\cwt_picture\CF\test_split\abnormal"

pattern = re.compile(r"0_(\d{3})")  # 匹配 0_ 后面接三个数字

# 遍历所有子文件夹
for folder_path, _, files in os.walk(root_folder):
    for filename in files:
        if filename.endswith(".npy"):
            # 如果文件名中匹配到了 0_xxx
            if pattern.search(filename):
                new_filename = pattern.sub(r"1_\1", filename)
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f"✅ 已改名: {filename} → {new_filename}")

print("🎉 全部文件名修改完成！")

