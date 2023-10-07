import torch.nn as nn
import torch
from torchvision import models

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.conv1 = nn.Conv2d(3, 1, 3) #input of size [N,C,H, W]
                                        #N==>batch size,
                                        #C==> number of channels,
                                        #H==> height of input planes in pixels,
                                        #W==> width in pixels.
        self.bowel = nn.Linear(64516, 1)
        self.extravasation = nn.Linear(64516, 1)
        self.kidney = nn.Linear(64516, 3)
        self.liver = nn.Linear(64516,3) 
        self.spleen = nn.Linear(64516, 3)
        
        #* activations ---->
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        
        x = self.conv1(x)
        # extract features
        x = torch.flatten(x, 1)
        
        # output logits
        bowel = self.bowel(x)
        
        extravsation = self.extravasation(x)
        
        kidney = self.kidney(x)
        
        liver = self.liver(x)
        
        spleen = self.spleen(x)

        output = torch.cat([self.sigmoid(bowel), self.sigmoid(extravsation), self.softmax(kidney), self.softmax(liver), self.softmax(spleen)], 1)
        
        return output
    
