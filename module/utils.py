import torch
import torchvision.transforms.functional as TF
import os
import numpy as np
import random
from numpy.random import RandomState

def evaluate_accuracy_ours(mw_model, test_loader, target_attr_idx, device, param1 = -1):
  mw_model.eval()
  mw_correct = 0
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)

        mw_outputs  = mw_model(data)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def evaluate_accuracy_simple(mw_model, test_loader, target_attr_idx, device, param1 = -1):
  mw_model.eval()
  mw_correct = 0
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)

        mw_outputs  = mw_model(data)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def write_to_file(filename, text):
    if not os.path.exists(filename):
       os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a') as f:
        f.write(text)
        f.write('\n')

def rotate_tensor(tensor, angle):
    tensor = TF.rotate(tensor, angle)
    return tensor

class mixup_data:
    def __init__(self, alpha = 0.8):
        self.alpha = None
        self.config(alpha)
    
    def config(self, alpha):
        self.alpha = alpha
    
    def __call__(self, x1, x2):
        # print("x1",x1[0])
        # print("x2",x2)
        x1 = x1 * self.alpha
        # print("x1",x1[0])
        x1 += x2 * (1 - self.alpha)
        # print("x1",x1[0])
        return x1

dic_functions = {
    'ours': evaluate_accuracy_ours,
    'Simple': evaluate_accuracy_simple,
    'set_seed': set_seed,
    'write_to_file': write_to_file,
    'rotate_tensor': rotate_tensor,
}
