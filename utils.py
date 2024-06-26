import torch
import torchvision.transforms.v2 as transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

def preprocess_img(path):
    img = Image.open(path)
 
    transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224)),  
    transforms.ToDtype(torch.float32, scale=True),           
])  
    out = transform(img).unsqueeze(0)
    return out

def get_imagenet_label(output, topk=False):
       
    pred_confidence, pred_idx = torch.max(output, dim=1, keepdim=True)
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    return pred_confidence, idx2label[pred_idx]

def display_images(img, adv_img, adv_pattern, preds, final_preds, eps, K=5):

    confidence, label = get_imagenet_label(preds)
    confidence_adv, label_adv = get_imagenet_label(final_preds)

    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].set_title('Input \n {} : {:.2f}% Confidence'.format(label,confidence.item()*100))
    ax[0].imshow(img.permute(1,2,0))
    ax[1].set_title('Adversarial Pattern')
    ax[1].imshow(adv_pattern.squeeze(0).permute(1,2,0))
    ax[2].set_title('Corrupted with epsilon = {:0.3f} \n {} : {:.2f}% Confidence'.
                    format(eps,label_adv,confidence_adv.item()*100)) 
    ax[2].imshow(adv_img.detach().permute(1,2,0))
    
    plt.show()
    plt.close()



