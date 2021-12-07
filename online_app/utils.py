from os import listdir
from os.path import isfile, join
import json
import csv
import pandas as pd
import numpy as np

DATA_PATH = "yelp_dataset/"


def read_data(name: str, return_df: bool = False, filter_condition: dict = None):
    data = []
    f = open(DATA_PATH + name + '.json', 'r', encoding='utf-8')
    count = 0
    for line in f:
        ifbreak = False
        if filter_condition is not None:
            oneline = json.loads(line)
            for key, value in filter_condition.items():
                if oneline[key] != value:
                    ifbreak = True
                    break
        if not ifbreak:
            data.append(json.loads(line))
            count += 1
    f.close()
    if return_df is True:
        return pd.DataFrame(data), count
    return data, count


def Attributes_flat(Attributes: dict, ID: dict = None):
    result = {}
    if Attributes is None:
        return result
    if ID is not None:
        result.update(ID)
    for key, value in Attributes.items():
        if type(value) is str and value[0] == '{':
            ldict = {}
            exec("dict_value = {}".format(value), globals(), ldict)
            dict_value = ldict['dict_value']
            result.update({key+'.'+second_key: second_value for second_key, second_value in Attributes_flat(dict_value).items()})
        else:
            result.update({key: value})
    return result