import torch
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
from utils import preprocess_img
from Attacks.FGSM import FGSM

# TO-DO: Implement as dataloader
def test(model, device, test_img):
    
    test_img = test_img.to(device)
    test_img.requires_grad = True

    output = model(test_img)
    pred_confidence, pred_idx = output.max(dim=1, keepdim=True)

    return pred

img = Image.open("./Images/flamingo.jpg")
img = preprocess_img(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

pred = test(model, device, img.unsqueeze(0))

#corrupted = FGSM(image=img, epsilon=, grads=)