import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
from sklearn.metrics import roc_auc_score
import preprocess as pp
import pandas as pd
from collections import defaultdict
from train import MolecularGraphNeuralNetwork, Trainer, Tester
from rdkit import Chem

import pandas as pd

# 1000    14849.786907899892      184799.68476867676      28.969444       271.78558
def mae(y_true, predictions): # https://stackoverflow.com/questions/74693070/how-can-i-calculate-the-mae-mean-absolute-error-in-pandas
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions)) 

#writes predictions to csv
def write_prediction_csv(method_name, predictions):
    with open(f"{method_name}.csv", 'w') as f:
        for line in predictions:
            f.write(f"{line}\n")

model = torch.load("../output/model--regression--1000iters.pck", weights_only=False)

trainer = Trainer(model)
tester = Tester(model)
test_dataset = pp.create_datasets(task="regression",dataset="melting", testing=True)
mae_val = tester.test_regressor(list(test_dataset),save_predictions=False)
print("mae val",mae_val)
preds, corrects = tester.test_regressor(list(test_dataset),save_predictions=True)
# for tup in zip(preds, corrects):
#     print(tup)
print("MAE for graph neural network ", mae(preds,corrects))
write_prediction_csv("gnn", preds)
write_prediction_csv("testing", corrects)
# test_csv = pd.read_csv("../dataset/regression/melting/test.csv")
# ids = test_csv["id"]
# ids_preds = zip(ids,preds)
# with open('submission_bradley.csv', 'w', newline='') as csvfile:
#     csvfile.write("id,Tm\n")
#     for tup in ids_preds:
#         csvfile.write(f"{tup[0]},{tup[1]}\n")

