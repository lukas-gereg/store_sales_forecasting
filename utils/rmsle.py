import torch

class RMSLE(torch.nn.Module):
    def __init__(self):
        super(RMSLE, self).__init__()
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        y_true = y_true + self.eps
        y_pred = torch.clamp(y_pred, min=0  + self.eps)  # Avoid negative predictions

        return torch.sqrt(torch.mean((torch.log(y_true + 1) - torch.log(y_pred + 1)) ** 2))