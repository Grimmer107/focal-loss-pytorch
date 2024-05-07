import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.alpha is not None:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
            probs = F.softmax(inputs, dim=1)
            
            self.alpha = torch.div(self.alpha, torch.sum(self.alpha))
            
            alpha_weight = self.alpha[targets].view(-1, 1)
            focal_weight = (1 - probs) ** self.gamma
            
            focal_loss = -alpha_weight * focal_weight * torch.log(probs)
            focal_loss = torch.sum(focal_loss * targets_one_hot, dim=1)
            
        else:
            probs = F.softmax(inputs, dim=1)
            focal_loss = -((1 - probs) ** self.gamma) * torch.log(probs)
            focal_loss = torch.sum(focal_loss * F.one_hot(targets, num_classes=inputs.size(1)).float(), dim=1)

        return focal_loss.mean()


if __name__ == "__main__":
    
    # example for five class classification
    logits = torch.rand(2, 5)
    targets = torch.tensor([1, 2])
    alpha = [0.12, 0.21, 0.15, 0.24, 0.27]
    
    # with alpha 
    fl = FocalLoss(alpha=alpha, gamma=2)
    print(fl.forward(logits, targets))

    # without alpha
    fl = FocalLoss(gamma=2)
    print(fl.forward(logits, targets))