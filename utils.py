import torch
import torchvision.transforms.v2 as transforms

import matplotlib.pyplot as plt
import json

def preprocess_img(img):
    transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224)),  
    transforms.ToDtype(torch.float32, scale=True), 
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])          
])
    return transform(img)

def get_imagenet_label(output):
    pred_confidence, pred_idx = output.max(dim=1, keepdim=True)
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    return pred_confidence, idx2label[pred_idx]

def display_images(image, corrupted, adv_pattern, predictions, final_predictions, eps):
  confidence, label = get_imagenet_label(predictions)
  confidence_adv, label_adv = get_imagenet_label(final_predictions)

  _, ax = plt.subplots(1, 3, figsize=(12, 4))
  ax[0].set_title('Input \n {} : {:.2f}% Confidence'.format(label,confidence.item()*100))
  ax[0].imshow(image.permute(1,2,0))
  ax[1].set_title('Adversarial Pattern')
  ax[1].imshow(adv_pattern.squeeze(0).permute(1,2,0))
  ax[2].set_title('Corrupted with epsilon = {:0.3f} \n {} : {:.2f}%Confidence'.
                  format(eps,label_adv,confidence_adv.item()*100)) 
  ax[2].imshow(corrupted.detach().permute(1,2,0))
  plt.tight_layout()
  plt.show()
