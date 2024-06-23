import torch

class Attack():
    def __init__(self, model, config):

        self.config = config
        self.model = model
        self.device = next(model.parameters()).device

    def random_initialization(self, img):
        random_init = torch.rand(img.size(), dtype=img.dtype, device=img.device)
        img = img + (random_init*self.config['eps'])
        img = torch.clamp(img,0,1)

        return img

    def __call__(self, img, label):

        img_adv = self.forward(img, label)
        return img_adv