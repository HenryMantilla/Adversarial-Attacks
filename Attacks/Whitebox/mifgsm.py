import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Attack import Attack
# MI-FGSM [https://arxiv.org/abs/1710.06081]

class MIFGSM(Attack):
    def __init__(self, model, config, target=None):
        super(MIFGSM, self).__init__(model, config)
        self.target = target

    def forward(self, img, label):

        img = img.detach().clone().to(self.device)

        adv_img = img.detach().clone()
        adv_img = self.random_initialization(adv_img)
        momentum = torch.zeros_like(adv_img, device=adv_img.device)

        for _ in range(self.config['iters']):

            adv_img.requires_grad = True
            pred = F.softmax(self.model(adv_img), dim=1)

            if self.target is None:
                label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
                loss = - nn.CrossEntropyLoss()(pred, label)
            else:
                self.target = torch.tensor(self.target, dtype=torch.long)
                loss = nn.CrossEntropyLoss()(pred, self.target)

            grad = torch.autograd.grad(loss, adv_img, retain_graph=False, create_graph=False)[0]
            grad /= torch.norm(grad, p=1)
            grad += momentum * self.config['decay']

            momentum = grad

            adv_pattern = img - self.config['alpha'] * grad.sign()
            #to satisfy l_infinity norm conditions
            adv_pattern = torch.clamp(adv_pattern - img, min=-self.config['eps'], max=self.config['eps']) 
            adv_img = torch.clamp(img + adv_pattern, min=0, max=1)


        return adv_img, grad
