"""Scrape saved "top rated games" page from steamdb.info for app data"""

from bs4 import BeautifulSoup
import pandas as pd

soup = BeautifulSoup(open("./data/ratings.html", "r"), 'html.parser')
apps = soup.select('tr.app')
res = [{"app_id": app.select('a[href]')[0].text,
        "app_name": app.select('td + td + td')[0].text,
        "pos": app.select('td.pos')[0].text,
        "neg": app.select('td.neg')[0].text,
        "wilson": app.select('td.neg + td')[0].text,
        "steam_score": app.select('td.neg + td + td')[0].text}
       for app in apps]
df = pd.DataFrame(res)
df.to_csv("./data/ratings.csv")
