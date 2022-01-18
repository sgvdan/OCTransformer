from attn_data import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils2 import *
from model_running import *
import numpy as np
import random
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2 as cv
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import io
from PIL import Image
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
from pytorch_grad_cam.ablation_layer import AblationLayerVit

seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="test_attn_plus_gradcam")

label_names = {
    0: "NORMAL",
    1: "CNV",
    2: "DME",
    3: "DRUSEN",
}
CLS2IDX = label_names


def reshape_transform(tensor, height=31, width=32):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * 0.4 + np.float32(img)
    cam = cam / np.max(cam)
    return cam


name = 'vit_base_patch16_224'
# initialize ViT pretrained
model_timm = timm.create_model(name, num_classes=4, img_size=(496, 512))
model_timm.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
model_timm = model_timm.to(device)

model_attn = vit_LRP(num_classes=4, img_size=(496, 512))
model_attn.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
model_attn = model_attn.to(device)
model_attn.eval()
attribution_generator = LRP(model_attn)

pytorch_total_params = sum(p.numel() for p in model_timm.parameters())
pytorch_total_params_train = sum(p.numel() for p in model_timm.parameters() if p.requires_grad)
wandb.log({"Total Params": pytorch_total_params})
wandb.log({"Trainable Params": pytorch_total_params_train})


def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).to(device),
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 31, 32)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(496, 512).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, transformer_attribution


def_args = dot_dict({
    "train": ["../../../data/kermany/train"],
    "val": ["../../../data/kermany/val"],
    "test": ["../../../../data/kermany/test"],
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
correct = 0.0
correct_arr = [0.0] * 10
total = 0.0
total_arr = [0.0] * 10
predictions = None
ground_truth = None
# Iterate through test dataset

columns = ["id", "Original Image", "Predicted", "Logits", "Truth", "Correct", "Attention NORMAL", "Attention CNV",
           "Attention DME",
           "Attention DRUSEN", "GradCAM", 'ScoreCAM', 'GradCAMPlusPlus', 'XGradCAM', 'EigenCAM', 'EigenGradCAM', 'Avg']
# for a in label_names:
#     columns.append("score_" + a)
test_dt = wandb.Table(columns=columns)

for i, (images, labels) in enumerate(test_loader):
    if i % 10 == 0:
        print(f'image : {i}\n\n\n')
    images = Variable(images).to(device)
    labels = labels.to(device)
    images = images.squeeze()
    # Forward pass only to get logits/output
    # print(images.shape)
    outputs_attn = model_attn(images.unsqueeze(0))
    outputs_timm = model_attn(images.unsqueeze(0))

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs_attn.data, 1)

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

    target_layers = [model_timm.blocks[-1].norm1]

    cams = [GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, EigenGradCAM]
    res = []
    just_grads = []
    images = images.unsqueeze(0)
    image_transformer_attribution = None
    for cam_algo in cams:
        # print(images.shape)
        cam = cam_algo(model=model_timm, target_layers=target_layers,
                       use_cuda=True if torch.cuda.is_available() else False, reshape_transform=reshape_transform,
                       )
        target_category = labels.item()
        grayscale_cam = cam(input_tensor=images, aug_smooth=True, eigen_smooth=True)
        just_grads.append(grayscale_cam[0, :])
        image_transformer_attribution = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
        vis = show_cam_on_image(image_transformer_attribution, grayscale_cam[0, :])
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        res.append(vis)  # superimposed_img / 255)
    gradcam = res
    images = images.squeeze()
    cat, attn_map = generate_visualization(images)
    attn_diff_cls = []
    for j in range(4):
        attn_diff_cls.append(generate_visualization(images, class_index=j)[0])
    avg = attn_map.copy() * 6
    # print(avg.max())
    for j, grad in enumerate(just_grads):
        g = grad.copy()
        # plt.imshow(g)
        # plt.title(str(j))
        # plt.show()
        g = np.where(g < g.max() / 4, g / 7, g)
        g = np.exp(g)
        g = g - g.min()
        g = g / g.max()
        # plt.imshow(g)
        # plt.title(str(j))
        # plt.show()
        avg += g
        # print(avg.max())
    avg = avg / avg.max()
    # plt.imshow(avg)
    # plt.show()
    vis = show_cam_on_image(image_transformer_attribution, avg)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    # plt.imshow(vis)
    # plt.show()
    avg = vis
    T = predicted.item() == labels.item()
    out = outputs_timm

    plt.bar(label_names, out.cpu().detach().numpy()[0])
    # plt.xlabel(label_names)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)

    row = [i, wandb.Image(images), label_names[predicted.item()], wandb.Image(im), label_names[labels.item()], T,
           wandb.Image(attn_diff_cls[0]), wandb.Image(attn_diff_cls[1]), wandb.Image(attn_diff_cls[2]),
           wandb.Image(attn_diff_cls[3]), wandb.Image(gradcam[4]), wandb.Image(gradcam[1]), wandb.Image(gradcam[2]),
           wandb.Image(gradcam[3]), wandb.Image(gradcam[4]), wandb.Image(gradcam[4]), wandb.Image(avg)]
    test_dt.add_data(*row)

    # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
    #                                                    y_true=ground_truth, preds=predictions,
    #                                                    class_names=label_names)})

accuracy = correct / total
metrics = {f'Test Accuracy_{name}': accuracy}
for label in range(4):
    metrics[f'Test Accuracy_{name}' + label_names[label]] = correct_arr[label] / total_arr[label]
wandb.log(metrics)
wandb.log({f"Grads_{name}": test_dt})
