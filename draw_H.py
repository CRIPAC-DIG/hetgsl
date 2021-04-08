import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import seaborn as sns
from dataset_utils import build_dataset, get_mask
import torch.nn.functional as F
import torch
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Chameleon')
args = parser.parse_args()

dataset = build_dataset(args.dataset, to_cuda=False)

x = dataset['features']
# normed_adj = dataset['normed_adj']
raw_adj = dataset['raw_adj']
y = dataset['labels']
y = F.one_hot(y) + 0.0
# pdb.set_trace()
ground_truth_H = (y.T @ raw_adj @ y )/ (y.T @ raw_adj @ torch.ones_like(y))

print(ground_truth_H)
ax = sns.heatmap(ground_truth_H, linewidth=0.5)
# plt.show()
plt.savefig(f'{args.dataset}_gtH.png')
plt.close()

filename = f'{args.dataset}_savedH.npy'
print(f'Loading {filename}')
with open(filename, 'rb') as f:
    saved_H = np.load(f)
    # plt.plot(saved_H)
print(saved_H)
ax = sns.heatmap(saved_H, linewidth=0.5)
# plt.show()
plt.savefig(f'{args.dataset}_savedH.png')
plt.close()