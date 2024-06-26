import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from utils import preprocess_img, display_images
from Attacks.Whitebox import FGSM

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights="IMAGENET1K_V1")
model.eval()

img_path = "./Images/flamingo.jpg"
img = preprocess_img(img_path)

config = {
    'eps': 0.15
}

attack = FGSM(model, config, target=None)
adv_img, adv_pattern = attack(img, label=130)

org_pred = F.softmax(model(img), dim=1) 
adv_pred = F.softmax(model(adv_img), dim=1)

display_images(img.squeeze(), adv_img.squeeze(), adv_pattern, org_pred, adv_pred, config['eps'])

