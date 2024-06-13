import numpy as np
import torch
from torch.nn import functional as F
from globals import *

def calculate_transfer_score(x, source_classifier, domain_discriminator):
    return source_classifier(x).max() + domain_discriminator(x)

def get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds):
    ws = torch.stack([calculate_transfer_score(x, source_classifier, domain_discriminator) for x in ut_features])
    if torch.isnan(ws).any():
        print("NaN found in ws in get source weights")
        
    # Debugging for w_alpha
    valid_ws = ws.reshape(-1, 1).flatten() >= w_alpha
    #print(f"ws: {ws}")
    print(f"w_alpha: {w_alpha}")
    #print(f"valid_ws: {valid_ws}")
    print(f"sum(valid_ws): {torch.sum(valid_ws)}")
    
    if torch.sum(valid_ws) == 0:
        print("No valid ws found above w_alpha")
        # Set V to all ones if no valid ws is found
        V = torch.ones_like(ut_preds[0])
    else:
        V = torch.sum(ut_preds[valid_ws], axis=0) / torch.sum(valid_ws).item()

    if torch.isnan(V).any():
        print("NaN found in V in get source weights")
    weights = torch.stack([V[label_idx] for label_idx in s_label])
    if torch.isnan(weights).any():
        print("NaN found in weights in get source weights")
    return weights

def predict(feature, source_classifier, domain_discriminator, prototype_classifier, w_0):
    w = calculate_transfer_score(feature, source_classifier, domain_discriminator)
    if w > w_0:
        return torch.argmax(F.softmax(source_classifier(feature)))
    else:
        return torch.argmax(F.softmax(prototype_classifier(feature)))