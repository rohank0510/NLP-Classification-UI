import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.examples import sentences
from spacy_streamlit import visualize_ner
import spacy_streamlit
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline
from statistics import mean
from st_aggrid.shared import JsCode

cellsytle_jscode = JsCode(
    """
function(params) {
    if (params.value.includes('Negative')) {
        return {
            'color': 'white',
            'backgroundColor': 'darkred'
        }
    } else if(params.value.includes('Positive')) {
        return {
            'color': 'black',
            'backgroundColor': 'lightgreen'
        }
    }
    else {
        return {
            'color': 'black',
            'backgroundColor': 'white'
        }
    }
};
"""
)


def load_sentiment_model():
    model = DistilBertForSequenceClassification.from_pretrained('training')
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased', num_labels=3)
    sentiment = pipeline("sentiment-analysis",
                         model=model, tokenizer=tokenizer)
    return sentiment

def get_score(text):
    label = sentiment(text)[0]['label']
    return 'Negative' if '0' in label else ('Postive' if '2' in label else 'Neutral')

def get_org_score(text):
    org_list = {}
    sentences = sentencizer(text)
    for sent in sentences.sents:
        doc = nlp(sent.text)
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                label = sentiment(text)[0]['label']
                int_label = -1 if '0' in label else (1 if '2' in label else 0)
                if ent.text in org_list:
                    org_list[ent.text].append(int_label)
                else:
                    org_list[ent.text] = [int_label]
    # return pd.DataFrame.from_dict({org: (mean(scores)-1) for org, scores in org_list.items()})
    return [{'org_name': org, 'score': convert_score_word(mean(scores))} for org, scores in org_list.items()]


def convert_score_word(score):
    return 'Negative' if score < 0 else 'Positive' if score > 0 else 'Neutral'

sentencizer = spacy.load("en_core_web_sm")
nlp = spacy.load('NER_CUSTOM_MODEL')
sentiment = load_sentiment_model()


f = open("testing_text.txt", "r")
text = f.read()


default_text = st.text_area("Message", height=100, value = text)

score = get_score(default_text)
df_score = get_org_score(default_text)
# print(df_score)
df = pd.read_csv('news_data_ROW 1 - 1330.csv')
pdf = df[['title', 'Sentimental Analysis']]

dict = {0: 'Neutral', -1:'Negative', 1:'Positive'}
pdf = pdf.replace({"Sentimental Analysis": dict})

st.title(' Sentiment Analysis')
st.subheader('Training Dataset with labels')
gb = GridOptionsBuilder.from_dataframe(pdf)
gb.configure_pagination(paginationPageSize = 30)
gb.configure_column("Sentimental Analysis", cellStyle=cellsytle_jscode)
gridOptions = gb.build()
AgGrid(pdf, gridOptions=gridOptions, theme='streamlit',
       editable=True, fit_columns_on_grid_load=True, enable_enterprise_modules=True, allow_unsafe_jscode=True)

st.subheader('Overall Sentiment Score')
st.text(score)
st.subheader('Detailed Sentiment Score')
st.dataframe(df_score)

doc2 = nlp(default_text)
visualize_ner(doc2, labels=nlp.get_pipe("ner").labels, key='ner',title="Named Entity Recognition")
