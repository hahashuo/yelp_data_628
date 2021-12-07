import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
import numpy as np


def wordcloud_by_freq(frequency: dict, outfile):
    wordcloud = WordCloud(background_color="white", width=1500, height=960, margin=10, max_words=50)
    wordcloud.generate_from_frequencies(frequencies=frequency)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    del wordcloud
    plt.close()


def word_score(data: pd.DataFrame, outfile):
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12, 9), sharex='row', sharey='row')
    W = ['service', 'table', 'time', 'lake', 'sunset', 'amazing', 'chicken', 'shrimp', 'ordered']
    for i, ax in zip(range(9), axes.flat):

        Score = []
        for index, row in data.iterrows():
            # print(row['c1'], row['c2'])
            if W[i] in row['text'].lower():
                Score.append(row["stars"])
        value = np.unique(Score, return_counts=True)[0]
        count = np.unique(Score, return_counts=True)[1]
        count = count / sum(count)
        ax.bar(value, count)
        ax.set_title(f"{W[i]}", fontsize=15)
    plt.savefig(outfile, format="svg", bbox_inches="tight")