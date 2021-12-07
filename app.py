import time

import pandas as pd
from flask import Flask, render_template, request, session, Response
from utils import read_data
from analysis import TfIdfCalculator, WordSegregator, first_text, second_text
from plots import wordcloud_by_freq, word_score
from io import BytesIO
import os


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = os.urandom(24)

GLOBAL_VAR = {}


@app.route('/')
def base():
    return render_template("base.html")


@app.route('/loading')
def loading():
    return render_template("loading.html")


@app.route('/search')
def search():
    keyword = request.args.get('keyword')
    keyword = keyword.lower()
    business, count = read_data('Wedding_business', return_df=True)
    count = 0
    result = []
    for name, address in zip(business.name, business.address):
        if keyword in name.lower():
            result.append({'name': name, 'address': address})
            count += 1
    if count == 0:
        return render_template("nosearchresult.html")
    elif count > 1:
        result_df = pd.DataFrame(result)
        return render_template("multiplesearch.html", tables=[result_df.to_html(classes='dataframe')],
                               titles=result_df.columns.values)
    this_one = business[business.name.apply(lambda x: keyword.lower() in x.lower())].iloc[0,:]
    GLOBAL_VAR['my_var'] = this_one.to_dict()
    Reviews, count = read_data("TOOLT", return_df=True)
    tf_idf_calculator = TfIdfCalculator()
    word_segregator = WordSegregator(only_nouns=True)
    token = list(map(word_segregator.transform, Reviews.text))
    InputText = pd.DataFrame({'text': token, 'id': Reviews.review_id})
    ResWeight = tf_idf_calculator.transform(InputText)
    Res_weight_df = pd.DataFrame(ResWeight).T
    Reviews['starofreviews'] = Reviews.stars
    Res_weight_df = pd.merge(Res_weight_df.reset_index(),
                             Reviews[['review_id', 'starofreviews']],
                             left_on='index', right_on='review_id')
    Res_weight_df = Res_weight_df.drop(['review_id', 'index'], axis=1)
    Star_Res = Res_weight_df.groupby('starofreviews').sum()
    GLOBAL_VAR['Star_Res'] = Star_Res
    GLOBAL_VAR['data'] = Reviews
    return render_template("report.html", first_text = first_text(this_one), second_text = second_text(this_one))


@app.route('/wordcloud.svg')
def word_cloud():

    star = request.args['star']
    Star_Res = GLOBAL_VAR['Star_Res']
    fake_file = BytesIO()
    wordcloud_by_freq(Star_Res.loc[int(star)].to_dict(), outfile=fake_file)
    return Response(fake_file.getvalue(),
                    headers={"Content-Type": "image/svg+xml"})

@app.route('/word_score.svg')
def words_score():
    fake_file = BytesIO()
    word_score(GLOBAL_VAR['data'], fake_file)
    return Response(fake_file.getvalue(),
                    headers={"Content-Type": "image/svg+xml"})


if __name__ == '__main__':
    app.run()
