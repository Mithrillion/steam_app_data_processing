import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd


ids = np.ravel(np.load("./data/ids_arr.npy"))
id_to_loc = {int(i): l for i, l in zip(ids, range(len(ids)))}
codes = np.load("./data/codes_arr.npy")
ratings = pd.read_csv("./data/ratings.csv")

tree = BallTree(codes)
loc = id_to_loc[221380]
dist, ind = tree.query([codes[loc]], k=20)
dist = np.ravel(dist)
ind = np.ravel(ind)

results = pd.DataFrame({"distance": dist, "loc": ind, "app_id": ids[ind].astype(np.int64)})
joined = pd.merge(left=results, right=ratings, on="app_id").sort_values("distance")\
    [["app_id", "distance", "app_name", "steam_score", "wilson"]]
print(joined)

