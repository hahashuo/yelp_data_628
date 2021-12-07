import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO


def wordcloud_by_freq(frequency: dict, outfile):
    wordcloud = WordCloud(background_color="white", width=1500, height=960, margin=10, max_words=30)
    wordcloud.generate_from_frequencies(frequencies=frequency)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(outfile, format="svg", bbox_inches="tight")
