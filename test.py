import torch
import torch.nn as nn
from torchvision.models import resnet34

from PIL import Image
from utils import preprocess_img, display_images
from Attacks.FGSM import generate_adv_img

import matplotlib.pyplot as plt
# TO-DO: Implement dataloader for images

criterion = nn.CrossEntropyLoss()

img = Image.open("./Images/flamingo.jpg")
img = preprocess_img(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet34(weights="IMAGENET1K_V1")
model.eval()

eps = 0.15
adv_img, adv_pattern, org_pred, final_pred = generate_adv_img(model, device, img.unsqueeze(0), epsilon=eps, target_label=130)
adv_img = adv_img.squeeze()

display_images(img, adv_img, adv_pattern, org_pred, final_pred, eps)

