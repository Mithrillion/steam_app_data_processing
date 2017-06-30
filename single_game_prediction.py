import numpy as np
import torch
import torch.nn.functional as F
from network import MultiImageAlexNet
import pickle
from PIL import Image
from torchvision import transforms
import collections
import pandas as pd
import os
import time


# image_dir = "./data/images"
image_dir = "./data/other_images"

class_map_file = "./data/class_maps/tags_map.pkl"

class_map = pickle.load(open(class_map_file, "rb"))
n_classes = len(class_map)

net = MultiImageAlexNet(num_classes=n_classes)

# load previous learning states
net.load_state_dict(torch.load("checkpoint.saved.pth.tar")['state_dict'])

net.cuda()
net.eval()


# temporarily copying the updated class definition from lastest pytorch/vision repo
class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


preprocess = transforms.Compose([
            Scale((356, 200)),
            transforms.ToTensor()
        ])


def predict_app(app_id):
    app_list = os.listdir(image_dir)
    if str(app_id) in app_list:
        curr_img_dir = os.path.join(image_dir, str(app_id))
        curr_img_names = os.listdir(curr_img_dir)
        if len(curr_img_names) <= 0:
            raise FileNotFoundError("Image files not found for the given app ID")
        else:
            img_tensors = []
            for img_name in curr_img_names:
                try:
                    img = Image.open(os.path.join(curr_img_dir, img_name))
                    tensor = preprocess(img)  # apply preprocessing
                    img_tensors.append(tensor.unsqueeze(0))
                except OSError:
                    continue
            if len(img_tensors) <= 0:
                raise FileNotFoundError("No valid images for the given app ID")
            else:
                img_inputs = img_tensors
                y = net.forward(img_inputs)
                y = F.sigmoid(y)
                pred = np.ravel(y.data.cpu().numpy())
                class_list = pd.DataFrame({'id': list(class_map.keys()),
                                           'tag': list(class_map.values()),
                                           'prob': pred})
    else:
        raise FileNotFoundError("Given app ID not found in data folder")
    return class_list

start_time = time.time()
list = predict_app(5).sort_values('prob', ascending=False)
print("execution time = {0}".format(time.time() - start_time))
print(list.head(20))
