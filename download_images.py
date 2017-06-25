import asyncio
from aiohttp import ClientSession
import os
from aiohttp import ClientConnectorError
import json
import re


async def get_image(image, session, directory, resolution="600x338", chunk_size=1<<15):
    url = image.replace("1920x1080", resolution)
    attempts = 0
    while attempts < 3:
        try:
            res = await session.get(url)
            filename = re.search("[\w|_|.]+\.jpg", url).group(0)
            with open(directory + filename, 'wb') as file:
                while True:  # save file
                    chunk = await res.content.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)
            return filename
        except ClientConnectorError:
            attempts += 1
            print("Connector error occurred!")
        except AttributeError:
            print("File {0} skipped!".format(url))
            return None
    if attempts == 3:
        return None


async def gather_results(images, directory):
    """Launch scrape tasks and collect results"""
    tasks = []
    async with ClientSession() as session:
        for image in images:
            task = asyncio.ensure_future(get_image(image, session, directory))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        # you now have all response bodies in this variable
        return responses


def process_done(future):
    """Save scrape results in json files"""
    cache =[res for res in future.result()]
    if len(cache) == 0:
        raise RuntimeError("Empty response!")
    else:
        print("processed {0} images".format(len(cache)))


start = 3500
step = 100
stop = 5000

for curr in range(start, stop, step):
    data = json.load(open("./data/scraped/scraped_{0}_{1}.json".format(curr, curr + step), "r"))
    i = 0
    for app_id, app_info in data.items():
        i += 1
        print("loading data for app: {0}, progress = {1}".format(app_id, i))
        directory = "./data/images/{0}/".format(app_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if app_info is not None:
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(gather_results(app_info["imgs"], directory))
            future.add_done_callback(process_done)
            loop.run_until_complete(future)

