import os
import torch
import numpy as np
from torch import nn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from shufflenet1d import ShuffleNetV2_1D  # 确保导入你的 1D 模型


def predict_folder_npy_1d(folder_path, model, class_indict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []
    embeddings = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".npy"):
            continue

        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)  # shape: (N,)

        # 转为 tensor 并加 batch & channel 维度 [1, 1, N]
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x)
            prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        # 从文件名提取真实标签 0_495.npy → 0
        true_label = int(file_name.split("_")[0])
        true_labels.append(true_label)
        predicted_labels.append(np.argmax(prob))
        embeddings.append(output.cpu().numpy().flatten())

    true_labels = np.array(true_labels)
    embeddings = np.array(embeddings)

    # t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i in range(len(class_indict)):
        idx = true_labels == i
        plt.scatter(embeddings_tsne[idx, 0], embeddings_tsne[idx, 1],
                    label=class_indict[str(i)], s=30)
    plt.legend()
    plt.title('t-SNE Visualization of 1D Feature Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(False)
    plt.show()

    # 混淆矩阵
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    num_classes = confusion_mat.shape[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion_mat[i, j]),
                     ha="center",
                     color="white" if confusion_mat[i, j] > confusion_mat.max() / 2 else "black")

    plt.colorbar()
    plt.title('Confusion Matrix (1D input)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(num_classes), [class_indict[str(i)] for i in range(num_classes)], rotation=45)
    plt.yticks(range(num_classes), [class_indict[str(i)] for i in range(num_classes)])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    folder_path = r"E:\BaiduNetdisk\Fault-diagnosis-based-on-deep-learning-main ResDSC\cwt_picture\CF\test_split"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 类别字典
    class_indict = {'0': 'normal', '1': 'abnormal'}

    # 加载 1D 模型
    model = ShuffleNetV2_1D(num_classes=2, input_channels=1)
    model_weight_path = "./weightss_1d/model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    predict_folder_npy_1d(folder_path, model, class_indict)
