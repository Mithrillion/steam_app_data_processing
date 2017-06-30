import pandas as pd

rat = pd.read_csv("./data/ratings.csv").iloc[:, 1:]
rat.to_json("./data/ratings.json", orient='records')
