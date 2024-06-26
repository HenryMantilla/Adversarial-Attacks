import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Attack import Attack
# Fast Gradient Sign Attack [https://arxiv.org/abs/1412.6572]

class FGSM(Attack):
    def __init__(self, model, config, target=None):
        super(FGSM, self).__init__(model, config)
        self.target = target
    
    def forward(self, img, label):

        adv_img = img.detach().clone().to(self.device)
        adv_img = self.random_initialization(adv_img)
        adv_img.requires_grad = True

        self.model.zero_grad()

        pred = F.softmax(self.model(adv_img), dim=1)

        if self.target is None:
            label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
            loss = - nn.CrossEntropyLoss()(pred, label)
        else:
            self.target = torch.tensor(self.target, dtype=torch.long)
            loss = nn.CrossEntropyLoss()(pred, self.target)

        #loss.backward()

        grad = torch.autograd.grad(loss, adv_img, retain_graph=False, create_graph=False)[0]
        adv_img = adv_img + self.config['eps'] * grad.sign()
        adv_img = torch.clamp(adv_img, min=0, max=1)

        return adv_img, grad.sign()