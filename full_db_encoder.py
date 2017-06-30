import numpy as np
import torch
from network import MultiImageAlexNet
import pickle
from data_utils import FullAppImageDataset
import torch.utils.data as td


end = 10000
image_dir = "./data/images"
json_dir = "./data/scraped"
class_map_file = "./data/class_maps/tags_map.pkl"

dat = FullAppImageDataset(image_dir, json_dir, end=end)
n_exmaples = len(dat)
class_map = pickle.load(open(class_map_file, "rb"))
n_classes = len(class_map)

data_loader = td.DataLoader(dat, num_workers=4, pin_memory=True)

net = MultiImageAlexNet(num_classes=n_classes)
net.load_state_dict(torch.load("checkpoint.saved.pth.tar")['state_dict'])
net.cuda()

ids = []
codes = []
n_total = len(data_loader)
n_processed = 0
for id, imgs in data_loader:
    y = net.encode(imgs)
    # codes.append([id, y.squeeze(0).data.cpu().numpy()])
    ids.append(id)
    codes.append(y.squeeze(0).data.cpu().numpy())
    n_processed += 1
    if n_processed % 100 == 0:
        print("processed {0}/{1}".format(n_processed, n_total))

ids_arr = np.array(ids)
codes_arr = np.array(codes)
np.save("./data/ids_arr", ids_arr)
np.save("./data/codes_arr", codes_arr)

