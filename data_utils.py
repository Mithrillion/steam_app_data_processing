import numpy as np
from PIL import Image
import torch
from torch.utils import data
import os
import json
from torchvision import transforms
import pickle
import collections

"""
steps:
1. load json of app metadata
2. for each app:
    1. load all images
    2. convert images to tensors
    3. concatenate all image tensors in a list
    4. get image tags and get them into list format
3. convert tags into indicator vectors
4. combine image tensors and tags into training pairs
5. construct data iterator

alternatively, generate one (image, tags) pair for each image (and later use voting to classify apps)
(to be implemented)
"""

start = 0
step = 100
end = 200
image_dir = "./data/images"
json_dir = "./data/scraped"
class_map_file = "./data/class_maps/genres_map.pkl"


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


class AppImageDataset(data.Dataset):
    """Define the image-tags(or genres) dataset"""
    def __init__(self, image_dir, json_dir, class_map_file=class_map_file, start=0, step=100, end=500):
        super(AppImageDataset, self).__init__()
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.start = start
        self.step = step
        self.end = end

        # define preprocessing steps: first downscale the images, then convert to tensor
        self.preprocess = transforms.Compose([
            Scale((356, 200)),
            transforms.ToTensor()
        ])

        # read the complete dict of tags for class label assignment later
        self.class_map = pickle.load(open(class_map_file, "rb"))
        self.rev_map = {v: k for k, v in self.class_map.items()}
        self.class_count = len(self.class_map)

        # read the complete list of apps for iterator indexing
        self.apps = []
        for curr in range(start, end, step):
            data = json.load(open(os.path.join(json_dir, "scraped_{0}_{1}.json".format(curr, curr + step)), "r"))
            for app_id, app_info in data.items():
                if app_info is not None and app_info['imgs'] is not None:
                    curr_img_dir = os.path.join(image_dir, app_id)
                    if os.path.exists(curr_img_dir):
                        curr_img_names = os.listdir(curr_img_dir)
                        if len(curr_img_names) > 0:  # if at least one image exists for this app
                            self.apps.append((app_id, app_info))

    def __getitem__(self, index):
        app_id, app_info = self.apps[index]
        img_tensors = []  # list to hold all image tensors of selected app
        tags = []  # list to hold names of all tags of selected app
        curr_img_dir = os.path.join(self.image_dir, app_id)
        curr_img_names = os.listdir(curr_img_dir)
        for img_name in curr_img_names:
            try:
                img = Image.open(os.path.join(curr_img_dir, img_name))
                tensor = self.preprocess(img)  # apply preprocessing
                img_tensors.append(tensor)
            except OSError:
                continue
        # get image tags / genres
        if app_info['genre'] is not None:
            tags = app_info['genre']
        tags = [self.rev_map[x] for x in tags]  # map tag names to tag ids
        tags_vec = np.zeros(self.class_count, dtype=np.float)
        tags_vec[[tags]] = 1  # convert to indicator vector
        return img_tensors, torch.from_numpy(tags_vec).type(torch.FloatTensor)

    def __len__(self):
        return len(self.apps)

# test data loader
# dat = AppImageDataset(image_dir, json_dir, end=500)
# loader = data.DataLoader(dat, num_workers=4)
# for i, ex in enumerate(loader):
#     print("number of images in example {0}: {1}".format(i, len(ex[0])))
#     if i > 100:
#         break
#

