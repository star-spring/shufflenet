import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import shufflenet_v2_x1_0

def predict_folder_images(folder_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = shufflenet_v2_x1_0(num_classes=4).to(device)
    # load model weights
    model_weight_path = "./weights/model-25.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

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

            # print prediction results
            print("Image: {}".format(img_name))
            for i, prob in enumerate(predict):
                print("Class: {:10}   Probability: {:.3}".format(class_indict[str(i)], prob))

            # display the image
            plt.imshow(img)
            plt.title("Image: {}".format(img_name))
            plt.show()

if __name__ == '__main__':
    folder_path = "E:/Data_set/predict"
    predict_folder_images(folder_path)
