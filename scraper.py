"""Scrape Steam store page for highlight image urls and genre/tags"""

from bs4 import BeautifulSoup
import asyncio
from aiohttp import ClientSession
import numpy as np
import pandas as pd
from functools import partial
from aiohttp import ClientConnectorError
import json
import re
from bs4 import NavigableString

cookies = {'birthtime': '283993201', 'mature_content': '1'}  # bypass age check

async def get_html_data(app_id, session):
    """Access the Steam store page for an app and retrieve image url and genre data"""
    url = "http://store.steampowered.com/app/{0}".format(app_id)
    attempts = 0
    while attempts < 3:
        try:
            res = await session.get(url)
            html = await res.text()
            soup = BeautifulSoup(html, 'html.parser')
            highlights = soup.select("div.highlight_strip_screenshot img")
            imgs_small = [img['src'] for img in highlights]
            imgs_large = [s.replace("116x65", "1920x1080")[:-13] for s in imgs_small]
            attrs = {"imgs": imgs_large}
            try:
                attrs["tags"] = [re.search("[\w| |\-]+", tag.text).group(0) for tag in soup.select("a.app_tag")]
            except AttributeError:
                attrs["tags"] = None
            try:
                attrs["name"] = [p.text for p in soup.select("div.apphub_AppName")]
            except AttributeError:
                attrs["name"] = None
            try:
                attrs["genre"] = []
                selected = soup.find_all('b', string='Genre:')[0]
                while selected.nextSibling.name != 'b':
                    selected = selected.nextSibling
                    if type(selected) == NavigableString or selected.name == 'br':
                        continue
                    else:
                        attrs["genre"] += [selected.text]
            except AttributeError:
                attrs["genre"] = None
            except IndexError:
                attrs["genre"] = None
            return app_id, attrs
        except ClientConnectorError:
            attempts += 1
            print("Connector error occurred!")
    if attempts == 3:
        return app_id, None

async def gather_results(curr, step, app_ids):
    """Launch scrape tasks and collect results"""
    tasks = []
    async with ClientSession(cookies=cookies) as session:
        for app_id in app_ids[curr : curr + step]:
            task = asyncio.ensure_future(get_html_data(app_id, session))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        # you now have all response bodies in this variable
        return responses


def process_df(future, curr, step):
    """Save scrape results in json files"""
    cache = {k: v for k, v in future.result()}
    if len(cache) == 0:
        raise RuntimeError("Empty response!")
    else:
        json.dump(cache, open("./data/scraped/scraped_{0}_{1}.json".format(curr, curr + step), "w"))


df = pd.read_csv("./data/ratings.csv")
app_ids = df.loc[:1000, "app_id"].astype(str)

start = 0
end = 1000
step = 100

for curr in range(start, end, step):
    print("loading data from {0} to {1}".format(curr, curr + step))
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(gather_results(curr, step, app_ids))
    future.add_done_callback(partial(process_df, curr=curr, step=step))
    loop.run_until_complete(future)
