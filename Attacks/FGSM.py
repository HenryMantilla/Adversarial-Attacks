import torch
import torch.nn as nn
import torch.nn.functional as F
# Fast Gradient Sign Attack described by Goodfellow et. al. in https://arxiv.org/pdf/1412.6572

def fgsm(image, epsilon, grads):

    sign_grads = grads.sign()
    corrupted_image = image + epsilon * sign_grads
    corrupted_image = torch.clamp(corrupted_image, min=0, max=1)

    return corrupted_image, sign_grads

def generate_adv_img(model, device, test_img, epsilon, target_label):
    target_label = torch.tensor(target_label, dtype=torch.long)
    test_img = test_img.to(device)
    test_img.requires_grad = True

    pred = F.softmax(model(test_img), dim=1)
    loss = nn.CrossEntropyLoss()(pred, target_label.unsqueeze(0))
    model.zero_grad()
    loss.backward()

    grads = test_img.grad.data
    adv_img, adv_pattern = fgsm(test_img, epsilon, grads)
    final_pred = F.softmax(model(adv_img), dim=1)

    return adv_img, adv_pattern, pred, final_pred