import torch
from torchvision import models

# resnet34 for example
model = models.resnet34(pretrained=True)
model.eval()
var = torch.ones((1, 3, 224, 224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet34.pt")
