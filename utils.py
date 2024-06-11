import numpy as np
import torch
from torch.nn import functional as F
from globals import *

def calculate_transfer_score(x, source_classifier, domain_discriminator):
    return source_classifier(x).max() + domain_discriminator(x)

def get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds):
    ws = torch.stack([calculate_transfer_score(x, source_classifier, domain_discriminator) for x in ut_features])
    V = torch.sum(ut_preds[ws.reshape(-1,1).flatten() >= w_alpha], axis=0) / torch.sum(ws >= w_alpha).item()
    weights = torch.stack([V[label_idx] for label_idx in s_label])
    return weights

def predict(feature, source_classifier, domain_discriminator, prototype_classifier, w_0):
    w = calculate_transfer_score(feature, source_classifier, domain_discriminator)
    if w > w_0:
        return torch.argmax(F.softmax(source_classifier(feature)))
    else:
        return torch.argmax(F.softmax(prototype_classifier(feature)))