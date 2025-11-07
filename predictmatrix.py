import os
import json

import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import numpy as np

from model import shufflenet_v2_x1_0

def predict_folder_images(folder_path, model, class_indict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    true_labels = []  # 存储真实标签
    predicted_labels = []  # 存储预测标签
    embeddings = []  # 存储特征向量

    # t-sne初始可视化函数  perplexity为30  且这个代码因为perplexity过大 没用发生聚类
    # def start_tsne(folder_path, class_indict):
    #     print("正在进行初始输入数据的 t-SNE 可视化...")
    #     images = []
    #     labels = []
    #     for img_name in os.listdir(folder_path):
    #         img_path = os.path.join(folder_path, img_name)
    #         if os.path.isfile(img_path):
    #             # load and transform the image
    #             img = Image.open(img_path)
    #             img_tensor = data_transform(img).unsqueeze(0).to(device)
    #             images.append(img_tensor.cpu().numpy().flatten())
    #             true_label = int(img_name.split("-")[1].split(".")[0])  # 文件名格式应为 "样本编号-类别标签.jpg"
    #             labels.append(true_label)
    #
    #     images = np.array(images)
    #     X_tsne = TSNE().fit_transform(images)
    #     plt.figure(figsize=(10, 10))
    #     for i, label in enumerate(labels):
    #         plt.scatter(X_tsne[i, 0], X_tsne[i, 1], label=class_indict[str(label)])
    #     plt.title('t-SNE Initial Visualization')
    #     plt.legend()
    #     plt.show()
    #
    # start_tsne(folder_path, class_indict)



    # iterate over images in the folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            # load and transform the image
            img = Image.open(img_path)
            img_tensor = data_transform(img).unsqueeze(0).to(device)

            # predict class
            with torch.no_grad():
                output = model(img_tensor)
                predict = torch.softmax(output, dim=1).squeeze().cpu().numpy()

            # store true and predicted labels
            true_label = int(img_name.split("-")[1].split(".")[0])  # 文件名格式应为 "样本编号-类别标签.jpg"
            true_labels.append(true_label)
            predicted_label = np.argmax(predict)
            predicted_labels.append(predicted_label)

            # store embeddings
            embeddings.append(output.cpu().numpy().flatten())

    #(绘制可视化分类图
    # convert lists to numpy arrays
    embeddings = np.array(embeddings)
    true_labels = np.array(true_labels)

    # reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # plot t-SNE visualization
    plt.figure(figsize=(8, 6))
    for i in range(len(class_indict)):
        indices = true_labels == i
        plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], label=class_indict[str(i)])
    plt.legend()
    plt.title('t-SNE Visualization of Image Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(False)  # plt.grid(False)Remove grid lines
    plt.show()
    #)


    # compute confusion matrix
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    # plot confusion matrix 标注混淆矩阵中的预测数值
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
    for i in range(len(class_indict)):
        for j in range(len(class_indict)):
            plt.text(j, i, str(confusion_mat[i, j]), horizontalalignment="center", color="white")

    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(len(class_indict)), [class_indict[str(i)] for i in range(len(class_indict))], rotation=45)
    plt.yticks(np.arange(len(class_indict)), [class_indict[str(i)] for i in range(len(class_indict))])
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_path = "E:/Data_set/predict"
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = shufflenet_v2_x1_0(num_classes=4).to(device)
    model_weight_path = "./weights/model-25.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    predict_folder_images(folder_path, model, class_indict)
