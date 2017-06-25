import os
from PIL import Image
import shutil

IMG_ROOT = "./data/images/"
target_size = (600, 338)

game_folders = os.listdir(IMG_ROOT)
for game in game_folders:
    print("processing game id {0}".format(game))
    game_root = os.path.join(IMG_ROOT, game)
    image_list = os.listdir(game_root)
    conv_root = os.path.join(game_root, "converted")
    if os.path.exists(conv_root):
        shutil.rmtree(conv_root)
    os.makedirs(conv_root)
    for img in image_list:
        if img[-4:] == '.jpg':
            old = Image.open(os.path.join(game_root, img))
            new = old.resize(target_size, Image.ANTIALIAS)
            new.save(os.path.join(conv_root, "CONVERTED_" + img))
