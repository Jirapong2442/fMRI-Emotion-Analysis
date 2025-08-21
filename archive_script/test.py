import vaderSentiment
import textblob
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel
from scipy.special import softmax
import pandas as pd
#load model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL)

train = pd.read_csv("C:/Users/NA/Saved Games/dataset/MELD.Raw/MELD.Raw/train/train_sent_emo.csv")
X_raw = train['Utterance']
y_raw = train['Sentiment']


#apply model output probability for each sentiment
all_scores = []
input_vector = []
for sentence in sentence_arr['text']:
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model_sentiment(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    all_scores.append(scores)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sentiment = []
for sentence in sentence_arr['text']:
    vs = analyzer.polarity_scores(sentence)
    sentiment.append(vs)