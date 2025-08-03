import timm
import torch.nn as nn
from config import CLASSES 

NUM_CLASSES = len(CLASSES)

def get_model(model_name, freeze_backbone=True):
    model = timm.create_model(model_name, pretrained=True)
    model.reset_classifier(NUM_CLASSES)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model
