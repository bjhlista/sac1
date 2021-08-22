# https://query1.finance.yahoo.com/v7/finance/download/SPY?period1=1628121600&period2=1628380800&interval=1d&events=history&includeAdjustedClose=true

# Press the green button in the gutter to run the script.
import datetime
from io import StringIO

import requests
import urllib.parse
import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo import UpdateOne

if __name__ == '__main__':

    client = MongoClient("mongodb://localhost:27019")
    db = client["history"]
    ohlc = db["tickers"]

    # t1 = 1628121600000
    # t2 = 1628380800000
    # dt = datetime.datetime.utcfromtimestamp(t1 / 1000)
    #
    # print(dt)
    #
    dt_obj1 = datetime.datetime.strptime("1993-01-29 00:00:00 +0000", "%Y-%m-%d %H:%M:%S %z")
    dt_obj2 = datetime.datetime.strptime("2021-08-08 00:00:00 +0000", "%Y-%m-%d %H:%M:%S %z")
    t1 = int(dt_obj1.timestamp())
    t2 = int(dt_obj2.timestamp())
    #
    # print(millis)
    #
    # # https://query1.finance.yahoo.com/v7/finance/download/
    headers = {'Host': 'query1.finance.yahoo.com', 'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'}
    https = 'https://query1.finance.yahoo.com/'
    url = 'v7/finance/download/SPY?'
    query = 'period1='+str(t1)+'&period2='+str(t2)+'&interval=1d&events=history&includeAdjustedClose=true'
    req_string = https + url + query
    print(req_string)
    response = requests.get(req_string, headers=headers)
    print(response.content)
    s_binary = response.content
    s_ascii = s_binary.decode("ascii")

    s_ascii_io = StringIO(s_ascii)
    updates = pd.read_csv(s_ascii_io)

    insert_time = datetime.datetime.utcnow()

    updates.columns = [x.lower().replace(" ", "_") for x in updates.columns]
    records = updates.to_dict("records")
    for record in records:
        record["t"] = "SPY"
        record["it"] = insert_time

    result = ohlc.insert_many(records)


    # get latest records.   for each update, if existing is not None and existing is different than update, existing.a = False, update.a = True, insert update

    if True:
        exit()

    original_file = open("/Users/bjhlista/Library/Mobile Documents/com~apple~CloudDocs/Wormhole/Investing/historical_data/SPY.csv")
    data = pd.read_csv(original_file, chunksize=500)
    for chunk in data:
        chunk.columns = [x.lower().replace(" ", "_") for x in chunk.columns]
        records = chunk.to_dict("records")
        for record in records:
            record["t"] = "SPY"
            record["it"] = insert_time

        result = ohlc.insert_many(records)


    client.close()
#    updates_file = "/Users/bjhlista/Downloads/SPY.csv"
#    updates_df = pd.read_csv(updates_file, header=None)




