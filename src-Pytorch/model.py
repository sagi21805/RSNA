import torch.nn as nn
import torch
from torchvision import models

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.bowel = nn.Linear(65536, 1)
        self.extravasation = nn.Linear(65536, 1)
        self.kidney = nn.Linear(65536, 3)
        self.liver = nn.Linear(65536,3) 
        self.spleen = nn.Linear(65536, 3)
        
        #* activations ---->
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
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
    
