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
import torch
import timm.models.vision_transformer
from sklearn.decomposition import PCA


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        """
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        """
        if self.dist_token is None:
            return x[:, 1:]
        else:
            return x[:, 2:]

    def forward(self, x):
        """
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
        """

        x = self.forward_features(x)
        return x


timm.models.vision_transformer.VisionTransformer = VisionTransformer

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

dino = True

test_dataset = Kermany_DataSet(def_args.test[0], size=(496, 496) if dino else (496, 512))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)
names = ["vit_base_patch16_224", "vit_base_patch32_384"]  # , "res50", "res101", "res152"]
models = []
if dino:
    model = timm.create_model('vit_base_patch32_224', pretrained=False, num_classes=4,
                              img_size=(496, 496))
    model.load_state_dict(torch.load(f'q_dino.pt', map_location=torch.device(device)))
    name = "dino"
    model = model.to(device)
    models.append(model)
else:
    for name in names:
        model = timm.create_model(name, num_classes=4, img_size=(496, 512))
        model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
        model = model.to(device)
        models.append(model)

for name, model in zip(names, models):
    embds = []
    colors = []
    # model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
    # model = model.to(device)
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    predictions = None
    ground_truth = None
    # Iterate through test dataset
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i % 10 == 0:
                print(f'image : {i}\n\n\n')
            images = Variable(images).to(device)
            labels = labels.to(device)
            # Forward pass only to get logits/output
            outputs = model(images)

            embds.append(outputs.flatten().cpu().detach().numpy())
            colors.append(labels.item())

    embds = np.array(embds)
    colors = np.array(colors)
    # pca = PCA(n_components=64, svd_solver='arpack').fit_transform(embds)
    embedding = umap.UMAP(n_components=3).fit_transform(embds)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
    plt.title(f'Feature Map of {name} Network 2  ')
    plt.show()
    plt.savefig(f'Feature Map of {name} Network 2  ')
    point_cloud = np.hstack([embedding, colors.reshape(-1, 1)])
    wandb.log({f"3D_UMAP222_FeatureMap_{name}": wandb.Object3D(point_cloud)})
