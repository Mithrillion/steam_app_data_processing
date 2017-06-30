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

grad_norm_clip = 10
end = 10000
image_dir = "./data/images"
json_dir = "./data/scraped"
class_map_file = "./data/class_maps/tags_map.pkl"


def save_checkpoint(state, is_best, filename='checkpoint.saved.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.saved.pth.tar')

dat = AppImageDataset(image_dir, json_dir, class_map_file, end=end)
n_exmaples = len(dat)

# np.random.seed(7777)  # ensure in all subsequent runs, the sets are fixed
# train_indices = np.random.choice(range(n_exmaples), int(n_exmaples * 0.8), replace=False)
# rest = set(range(n_exmaples)).difference(set(train_indices))
# val_indices = np.random.choice(list(rest), int(n_exmaples * 0.05), replace=False)
# test_indices = list(rest.difference(set(val_indices)))
# pickle.dump((train_indices, val_indices, test_indices), open("./data/set_indices.pkl", "wb"))

train_indices, val_indices, test_indices = pickle.load(open("./data/set_indices.pkl", "rb"))

train_sampler = sampler.SubsetRandomSampler(train_indices)
val_sampler = sampler.SubsetRandomSampler(val_indices)
test_sampler = sampler.SubsetRandomSampler(train_indices)

train_loader = td.DataLoader(dat, num_workers=4, sampler=train_sampler, pin_memory=True)
val_loader = td.DataLoader(dat, num_workers=4, sampler=val_sampler, pin_memory=True)
# test_loader = td.DataLoader(dat, num_workers=4, sampler=test_sampler, pin_memory=True)

class_map = pickle.load(open(class_map_file, "rb"))
n_classes = len(class_map)

net = MultiImageAlexNet(num_classes=n_classes)

# # initialise
# for p in net.parameters():
#     p.data.normal_(0, 1e-2)
# for p in net.gru.parameters():
#     p.data.normal_(0, 1e-1)
#
# # load pretrained features
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }
# ref_net = AlexNet(num_classes=1000)
# ref_net.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
# net.features.load_state_dict(ref_net.features.state_dict())
# del ref_net

# load previous learning states
net.load_state_dict(torch.load("checkpoint.saved.pth.tar")['state_dict'])


net.cuda()

# test training
opt = optim.Adam(net.parameters(), 1e-4)
epochs = 20
batch = 16
n_log_batches = 4
s = 0
validation_interval = 1000
cum_loss = Variable(torch.zeros(1).cuda())
net.zero_grad()
for k in range(epochs):
    print("epoch = {0}".format(k))
    for imgs, target in train_loader:
        net.train()
        y = net.forward(imgs)
        y = F.sigmoid(y)
        loss = nn.BCELoss()(y, Variable(target.cuda(async=True)))
        cum_loss += loss
        s += 1
        if s % batch == 0:
            if s % (batch * n_log_batches) == 0:
                print("current loss = {0}".format(cum_loss.data.cpu().numpy()[0] / batch))
            cum_loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), grad_norm_clip)
            opt.step()
            cum_loss = Variable(torch.zeros(1).cuda())
            net.zero_grad()
        if s % validation_interval == 0:
            losses = torch.zeros(1).cuda()
            net.eval()
            preds = []
            targets = []
            for i, t in val_loader:
                y = net.forward(i)
                y = F.sigmoid(y)
                preds.append(y.data.cpu().numpy() > 0.5)
                targets.append(t.numpy())
                losses += nn.MSELoss()(y, Variable(t.cuda(async=True))).data
            val_loss = losses.cpu().numpy()[0] / len(val_loader)
            print("validation loss = {0}".format(val_loss))
            preds = np.concatenate(preds, 0)
            targets = np.concatenate(targets, 0)
            score = jaccard_similarity_score(targets, preds)
            r_score = label_ranking_average_precision_score(targets, preds)
            print("validation jaccard score = {0}".format(score))
            print("validation ranking score = {0}".format(r_score))
            save_checkpoint({
                'epoch': k,
                'steps': s,
                'state_dict': net.state_dict(),
                'val_loss': val_loss
            }, False)
