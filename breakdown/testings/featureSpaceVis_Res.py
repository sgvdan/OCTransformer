from data_for_tests import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils import *
from res_models import *
from model_running import *
from convnext import convnext_base, convnext_large, convnext_xlarge
import numpy as np
import random
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2 as cv
import cv2
import umap

wandb.init(project="featureViz")

seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def_args = dot_dict({
    "train": ["../../../data/kermany/train"],
    "val": ["../../../data/kermany/val"],
    "test": ["../../../data/kermany/test"],
})

label_names = [
    "NORMAL",
    "CNV",
    "DME",
    "DRUSEN",
]
test_dataset = Kermany_DataSet(def_args.test[0])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)
names = ["res18", "res50", "res101", "res152"]
models = [Resnet18(4), Resnet50(4), Resnet101(4), Resnet152(4)]
for name, model in zip(names, models):
    embds = []
    colors = []
    model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
    model = model.to(device)
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    predictions = None
    ground_truth = None
    # Iterate through test dataset

    columns = ["id", "Original Image", "Predicted", "Truth", "GradCAM", 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM',
               'XGradCAM', 'EigenCAM', 'FullGrad']
    # for a in label_names:
    #     columns.append("score_" + a)

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.resnet.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    print(f"Total convolutional layers: {counter}")

    for i, (images, labels) in enumerate(test_loader):
        if i % 10 == 0:
            print(f'image : {i}\n\n\n')
        images = Variable(images).to(device)
        labels = labels.to(device)
        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)
        correct += (predicted == labels).sum()

        for label in range(4):
            correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
            total_arr[label] += (labels == label).sum()

        if i == 0:
            predictions = predicted
            ground_truth = labels
        else:
            predictions = torch.cat((predictions, predicted), 0)
            ground_truth = torch.cat((ground_truth, labels), 0)

            accuracy = correct / total

        # pass the image through all the layers
        results = [conv_layers[0](images)]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = [results[-1]]
        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)
        for num_layer in range(len(outputs)):
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            embds.append(layer_viz.data.flatten().cpu().detach().numpy())
            colors.append(labels.item())
            # plt.figure(figsize=(30, 30))
            # print(layer_viz.size())
            # for i, filter in enumerate(layer_viz):
            #     if i == 64:  # we will visualize only 8x8 blocks from each layer
            #         break
            #     plt.subplot(8, 8, i + 1)
            #     plt.imshow(filter, cmap='gray')
            #     plt.axis("off")
            # print(f"Saving layer {num_layer} feature maps...")
            # plt.show()

    embds = np.array(embds)
    colors = np.array(colors)
    embedding = umap.UMAP(n_components=3).fit_transform(embds)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
    # plt.gca().legend(tuple(label_names))
    plt.title(f'Feature Map of {name} Network 2_')
    plt.show()
    plt.savefig(f'Feature Map of {name} Network 2_')
    plt.close()

    point_cloud = np.hstack([embedding, colors.reshape(-1, 1)])
    wandb.log({f"3D_UMAP_FeatureMap_{name}": wandb.Object3D(point_cloud)})
    metrics = {f'Test Accuracy_{name}': accuracy}
    for label in range(4):
        metrics[f'Test Accuracy_{name}' + label_names[label]] = correct_arr[label] / total_arr[label]
    wandb.log(metrics)
