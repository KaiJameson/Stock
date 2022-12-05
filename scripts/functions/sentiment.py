from statistics import mean
from functions.trade import get_api
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
import datetime
import copy
import sys


def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

def get_news(symbol, api):
    news = api.get_news(symbol, limit=99999)
    print(f"{len(news)} news articles were found for {symbol}")

    return news


def vad_sentiment(name, df, symbol):
    df_copy = copy.copy(df)

    api = get_api()
    news = get_news(symbol, api)
    
    # nltk.download('vader_lexicon')

    new_words = {
        'down': -1.0,
        'downgrade': -1.0,
        'downgrades': -1.0,
        'downgrading': -1.0,
        'fall': -1.0,
        'tumbles': -1.5,
        'bearish': -1.0,
        'losers': -1.0,
        'litigation': -.5,
        'weak': -.8,
        'cut': -1.0,
        'correction': -.5,
        'drops': -1.0,
        'illegal': -.5,
        'illegally': -.5,
        'disappointing': -.8,
        'short': -1.0,

        'strong': .8,
        'bargain': 1.0,
        'winners': 1.0,
        'monster': 1.0,
        'spikes': 1.0,
        'bullish': 1.0,
        'watch': .2,
        'watching': .2,
        'raises': 1.0,
        'higher': 1.0,
        'up': 1.0,
        'upgrade': 1.0,
        'upgrades': 1.0,
        'upgrading': 1.0,
        'surge': 1.5,
        "buyback": .8,
    }

    vader = SentimentIntensityAnalyzer()
    
    if name == "fin_vad":
        vader.lexicon.update(new_words)
        pop_words = ["alert"]
        for word in pop_words:
            vader.lexicon.pop(word)

    sent_list = []

    for ele in reversed(news):
        sent_list.append([ele.created_at.date(), vader.polarity_scores(ele.headline)["compound"]])

    properly_group_sentiment(sent_list, name, df, df_copy)


def fin_bert_sentiment(name, df, symbol):
    df_copy = copy.copy(df)

    api = get_api()
    news = get_news(symbol, api)
    
    sent_list = []

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


    for ele in reversed(news):
        input = tokenizer(ele.headline, padding = True, truncation = True, return_tensors='pt')
        # print(input)
        outputs = model(**input)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        # print(predictions.detach().numpy()[0], ele.headline, ele.created_at.date())
        # print(predictions.shape)

        # print(ele.created_at.date(), predictions[0], predictions[1], predictions[2], ele.headline)
        sent_list.append([ele.created_at.date(), predictions[0], predictions[1], predictions[2]])

    # print("", flush=True)
    sent_list = np.array(sent_list)

    # print(sent_list)
    # print(sent_list)
    # print(sent_list.shape)
    # print(sent_list[:, [0, 1]])

    # pd.set_option("display.max_rows", None)
    # print(sent_list[:, 0], flush=True)
    properly_group_sentiment(sent_list[:, [0, 1]], "fin_bert_pos", df, df_copy)
    properly_group_sentiment(sent_list[:, [0, 2]], "fin_bert_neg", df, df_copy)
    properly_group_sentiment(sent_list[:, [0, 3]], "fin_bert_neu", df, df_copy)


    # print(df.tail(60))

def properly_group_sentiment(sent_list, name, df, df_copy):

    sen_df = pd.DataFrame(sent_list, columns= ['date', name])
    cols = sen_df.columns 

    sen_df[cols[1:]] = sen_df[cols[1:]].apply(pd.to_numeric, errors='coerce')

    mean_scores = sen_df.groupby(['date']).mean()
    mean_scores = mean_scores[mean_scores.index >= df_copy.index[0]]


    for day in mean_scores.index:
        if day in df_copy.index:
            pass
        else:
            fit = False
            tmp_day = day
            while not fit:
                if tmp_day in df_copy.index:
                    mean_scores = mean_scores.rename(index={day:tmp_day})
                    fit = True
                else:
                    tmp_day -= datetime.timedelta(1)


    mean_scores = mean_scores.groupby([mean_scores.index]).mean()

    df_copy = df_copy.join(mean_scores, how="left")
    df_copy = df_copy.fillna(0.000000)
    df[name] = df_copy[name]


def fin_bert_return(name, df, symbol):
    pass

