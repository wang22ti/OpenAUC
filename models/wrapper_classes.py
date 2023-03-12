import torch
from torch import nn

class TimmResNetWrapper(nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x, return_features=True, dummy_label=None):
        x = self.resnet.forward_features(x)
        embedding = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        preds = self.resnet.fc(embedding)

        return embedding, preds if return_features else preds