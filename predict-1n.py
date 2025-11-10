import os
import json
import torch
import numpy as np
from tqdm import tqdm

from shufflenet1d import ShuffleNetV2_1D  # 引入你的 1D 模型


def main():
    # ---------------- 参数配置 ----------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_folder = r"E:\BaiduNetdisk\Fault-diagnosis-based-on-deep-learning-main ResDSC\cwt_picture\CF\test_split"
    model_weight_path = "./weightss_1d/model-27.pth"  # 修改为你训练保存的模型路径
    class_json_path = "./class_indices.json"  # 类别对应关系（比如 { "0": "normal", "1": "abnormal" }）

    # ---------------- 读取类别索引 ----------------
    assert os.path.exists(class_json_path), f"file '{class_json_path}' does not exist."
    with open(class_json_path, "r") as f:
        class_indict = json.load(f)

    # ---------------- 加载模型 ----------------
    model = ShuffleNetV2_1D(num_classes=2, input_channels=1).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # ---------------- 遍历测试集 ----------------
    assert os.path.exists(test_folder), f"Test folder '{test_folder}' does not exist."
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".npy")]
    print(f"共检测到测试样本: {len(test_files)} 个")

    results = []  # 保存预测结果

    with torch.no_grad():
        for file_path in tqdm(test_files, desc="Predicting"):
            data = np.load(file_path)
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, N]

            output = model(data)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = class_indict[str(pred_idx)]
            pred_prob = probs[pred_idx]

            results.append((os.path.basename(file_path), pred_label, float(pred_prob)))

    # ---------------- 输出结果 ----------------
    print("\n预测结果（前 10 条）:")
    for name, label, prob in results[:20]:
        print(f"{name:25} → {label:10} ({prob:.3f})")

    # ---------------- 保存结果到 CSV ----------------
    import pandas as pd
    save_path = "./test_predictions.csv"
    df = pd.DataFrame(results, columns=["filename", "pred_label", "confidence"])
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 所有预测结果已保存到: {save_path}")


if __name__ == '__main__':
    main()
