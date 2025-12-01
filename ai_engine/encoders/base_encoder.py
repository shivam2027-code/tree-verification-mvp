from torchvision import models, transforms
import torch.nn as nn
import torch

class BaseEncoder:
    def __init__(self):
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Identity()  # remove final layer
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def encode(self, image):
        img = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            vec = self.model(img)
        return vec.squeeze().numpy()
