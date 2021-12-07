import pandas as pd

import utils
from analysis import TfIdfCalculator, WordSegregator
from plots import wordcloud_by_freq

if __name__ == '__main__':
    Reviews, count = utils.read_data("TOOLT", return_df=True)
    TfIdfCalculator = TfIdfCalculator()
    WordSegregator = WordSegregator(only_nouns=True)
    token = list(map(WordSegregator.transform, Reviews.text))
    InputText = pd.DataFrame({'text': token, 'id': Reviews.review_id})
    ResWeight = TfIdfCalculator.transform(InputText)
    Res_weight_df = pd.DataFrame(ResWeight).T
    Reviews['starofreviews'] = Reviews.stars
    Res_weight_df = pd.merge(Res_weight_df.reset_index(),
                             Reviews[['review_id', 'starofreviews']],
                             left_on='index', right_on='review_id')
    Res_weight_df = Res_weight_df.drop(['review_id', 'index'], axis=1)
    Star_Res = Res_weight_df.groupby('starofreviews').sum()
    wordcloud_by_freq(Star_Res.loc[5].to_dict())