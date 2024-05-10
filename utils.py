import torch
import torchvision.transforms.v2 as transforms

def preprocess_img(img):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),           
])
    return transform(img)