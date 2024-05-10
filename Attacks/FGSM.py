import torch

# Fast Gradient Sign Attack described by Goodfellow et. al. in https://arxiv.org/pdf/1412.6572

def FGSM(image, epsilon, grads):

    sign_grads = grads.sign()
    corrupted_image = image + epsilon * sign_grads
    corrupted_image = torch.clamp(corrupted_image, min=0, max=1)

    return corrupted_image