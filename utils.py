import numpy as np
import torch
from torch.nn import functional as F

def calculate_transfer_score(x, source_classifier, domain_discriminator):
    return max(source_classifier(x))+domain_discriminator(x)

def predict(feature, source_classifier, domain_discriminator, prototype_classifier, w_0):
    w = calculate_transfer_score(feature, source_classifier, domain_discriminator)
    if w > w_0:
        return torch.argmax(F.softmax(source_classifier(feature)))
    else:
        return torch.argmax(F.softmax(prototype_classifier(feature)))