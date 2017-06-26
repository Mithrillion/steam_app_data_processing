"""
Generate the list of unique genres and tags for classification labelling
"""

import json
import os
import pickle

start = 0
step = 100
end = 10000
json_dir = "./data/scraped"
save_dir = "./data/class_maps"

genres = set()
tags = set()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for curr in range(start, end, step):
    data = json.load(open(os.path.join(json_dir, "scraped_{0}_{1}.json".format(curr, curr + step)), "r"))
    for app_id, app_info in data.items():
        if app_info is not None:
            app_genre = app_info['genre']
            app_tags = app_info['tags']
            if app_genre is None:
                app_genre = set()
            else:
                app_genre = set(app_genre)
            if app_tags is None:
                app_tags = set()
            else:
                app_tags = set(app_tags)
            genres = genres.union(app_genre)
            tags = tags.union(app_tags)

genres_map = dict(enumerate(genres))
tags_map = dict(enumerate(tags))
pickle.dump(genres_map, open(os.path.join(save_dir, "genres_map.pkl"), 'wb'))
pickle.dump(tags_map, open(os.path.join(save_dir, "tags_map.pkl"), 'wb'))
