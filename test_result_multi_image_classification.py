import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import MultiImageAlexNet, AlexNet
from data_utils import AppImageDataset
import torch.utils.data as td
import torch.utils.data.sampler as sampler
from sklearn.metrics import jaccard_similarity_score, label_ranking_average_precision_score
import pickle
from torch.utils import model_zoo
import shutil

end = 10000
image_dir = "./data/images"
json_dir = "./data/scraped"
class_map_file = "./data/class_maps/tags_map.pkl"

dat = AppImageDataset(image_dir, json_dir, class_map_file, end=end)
n_exmaples = len(dat)

# np.random.seed(7777)  # ensure in all subsequent runs, the sets are fixed
# train_indices = np.random.choice(range(n_exmaples), int(n_exmaples * 0.8), replace=False)
# rest = set(range(n_exmaples)).difference(set(train_indices))
# val_indices = np.random.choice(list(rest), int(n_exmaples * 0.05), replace=False)
# test_indices = list(rest.difference(set(val_indices)))
# pickle.dump((train_indices, val_indices, test_indices), open("./data/set_indices.pkl", "wb"))

train_indices, val_indices, test_indices = pickle.load(open("./data/set_indices.pkl", "rb"))

test_sampler = sampler.SubsetRandomSampler(train_indices)
test_loader = td.DataLoader(dat, num_workers=4, sampler=test_sampler, pin_memory=True)

class_map = pickle.load(open(class_map_file, "rb"))
n_classes = len(class_map)

net = MultiImageAlexNet(num_classes=n_classes)

# load previous learning states
net.load_state_dict(torch.load("checkpoint.saved.pth.tar")['state_dict'])


net.cuda()

# test result
batch = 16

preds = []
targets = []
losses = torch.zeros(1).cuda()
count = 0
for i, t in test_loader:
    y = net.forward(i)
    y = F.sigmoid(y)
    preds.append(y.data.cpu().numpy() > 0.5)
    targets.append(t.numpy())
    losses += nn.MSELoss()(y, Variable(t.cuda(async=True))).data
    count += 1
    if count % 200 == 0:
        print("processed {0}/{1}".format(count, len(test_loader)))
val_loss = losses.cpu().numpy()[0] / len(test_loader)
print("test loss = {0}".format(val_loss))
preds = np.concatenate(preds, 0)
targets = np.concatenate(targets, 0)
score = jaccard_similarity_score(targets, preds)
r_score = label_ranking_average_precision_score(targets, preds)
print("test jaccard score = {0}".format(score))
print("test ranking score = {0}".format(r_score))


