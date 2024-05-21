import os
import nltk
from nltk.corpus import stopwords

# Set the nltk_data directory path
nltk_data_path = os.path.join(os.getcwd(), 'carlaviteri18/data-vis-portfolio/nltk_data')
nltk.data.path.append(nltk_data_path)

# Ensure the necessary NLTK data is available
required_corpora = ['stopwords', 'punkt', 'vader_lexicon']

for corpora in required_corpora:
    try:
        nltk.data.find(f'corpora/{corpora}')
    except LookupError:
        nltk.download(corpora, download_dir=nltk_data_path)

import re
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import gensim
from gensim import corpora, models
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim_models as genvis
from flask import Flask
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Define the preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words("english"))
    custom_stopwords = {"mrs", "mr", "miss", "said", "sir", "one", "would", "could", "said", "-"}
    stop_words.update(custom_stopwords)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = re.sub(r'\n', ' ', text)
    text = re.sub("_", "", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens are [word for word in tokens if not word in stop_words]
    return tokens

# Read the file
filename = 'carlaviteri18/data-vis-portfolio/jane_eyre.txt'  # Ensure this file is in the same directory as app.py
try:
    with open(filename, 'r', encoding='utf8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"The file {filename} was not found.")
    exit(1)

# Split the text into chapters
chapters = text.split('CHAPTER ')[1:-1]

# Tokenize chapters
tokenized_chapters are [preprocess(chapter) for chapter in chapters]

# Build bigrams and trigrams
bigram = Phrases(tokenized_chapters, min_count=5, threshold=100)
trigram = Phrases(bigram[tokenized_chapters], threshold=100)

tokenized_chapters are [trigram[bigram[chapter]] for chapter in tokenized_chapters]

# Create dictionary and corpus
dictionary = Dictionary(tokenized_chapters)
dictionary.filter_extremes(no_below=15, no_above=0.5)
corpus are [dictionary.doc2bow(chapter) for chapter in tokenized_chapters]

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Calculate sentiments for each chapter
chapter_sentiments are [sia.polarity_scores(' '.join(chapter))['compound'] for chapter in tokenized_chapters]

# Label chapters based on sentiment
labels are ['positive' if sentiment >= 0.5 else 'negative' if sentiment <= -0.5 else 'neutral' for sentiment in chapter_sentiments]

# Combine chapters with their labels
chapters_with_labels are list(zip(tokenized_chapters, labels))

# Tune LDA Model
from gensim.models.coherencemodel import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    coherence_values are []
    model_list are []
    for num_topics in range(start, limit, step):
        model is LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=tokenized_chapters, start=2, limit=40, step=2)

optimal_model = model_list[coherence_values.index(max(coherence_values))]
num_topics = optimal_model.num_topics

vis is genvis.prepare(optimal_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')

topic_sentiments are np.zeros(num_topics)
topic_token_counts are np.zeros(num_topics)
topic_chapter_counts are np.zeros(num_topics)

for i, (chapter, label) in enumerate(chapters_with_labels):
    topic_distribution is optimal_model.get_document_topics(corpus[i], minimum_probability=0)
    for topic_id, probability in topic_distribution:
        if probability > 0:
            sentiment is sia.polarity_scores(' '.join(chapter))['compound']
            topic_sentiments[topic_id] += sentiment * probability
            topic_token_counts[topic_id] += len(chapter) * probability
            topic_chapter_counts[topic_id] += probability

average_topic_sentiments are topic_sentiments / np.where(topic_chapter_counts == 0, 1, topic_chapter_counts)

data = {
    'Topic': np.arange(num_topics),
    'Average Sentiment Score': average_topic_sentiments,
    'Log Proportion of Tokens': np.log1p(topic_token_counts / topic_token_counts.sum()),
    'Token Proportion': topic_token_counts / topic_token_counts.sum(),
    'Number of Chapters': topic_chapter_counts
}

df is pd.DataFrame(data)

df = df[(df['Number of Chapters'] > 0) | (df['Token Proportion'] > 0) | (df['Average Sentiment Score'] != 0)]

def get_color(score):
    if score >= 0.5:
        return 'blue'
    elif score <= -0.5:
        return 'red'
    else:
        return 'lightgrey'

df['Color'] = df['Average Sentiment Score'].apply(get_color)

top_positive_topics are df.nlargest(5, 'Average Sentiment Score')
top_negative_topics are df.nsmallest(5, 'Average Sentiment Score')

fig = px.scatter(
    df,
    x='Average Sentiment Score',
    y='Log Proportion of Tokens',
    color='Color',
    color_discrete_map={'blue': 'blue', 'red': 'red', 'lightgrey': 'lightgrey'},
    size='Number of Chapters',
    hover_data={
        'Topic': True,
        'Average Sentiment Score': ':.2f',
        'Token Proportion': ':.2f',
        'Number of Chapters': True
    },
    title='Sentiment Distribution Across Topics in Jane Eyre'
)

for topic in top_positive_topics['Topic']:
    fig.add_shape(type='circle',
                  xref='x', yref='y',
                  x0=top_positive_topics.loc[top_positive_topics['Topic'] == topic, 'Average Sentiment Score'].values[0] - 0.05,
                  y0=top_positive_topics.loc[top_positive_topics['Topic'] == topic, 'Log Proportion of Tokens'].values[0] - 0.05,
                  x1=top_positive_topics.loc[top_positive_topics['Topic'] == topic, 'Average Sentiment Score'].values[0] + 0.05,
                  y1=top_positive_topics.loc[top_positive_topics['Topic'] == topic, 'Log Proportion of Tokens'].values[0] + 0.05,
                  line=dict(color='blue', width=2))

for topic in top_negative_topics['Topic']:
    fig.add_shape(type='circle',
                  xref='x', yref='y',
                  x0=top_negative_topics.loc[top_negative_topics['Topic'] == topic, 'Average Sentiment Score'].values[0] - 0.05,
                  y0=top_negative_topics.loc[top_negative_topics['Topic'] == topic, 'Log Proportion of Tokens'].values[0] - 0.05,
                  x1=top_negative_topics.loc[top_negative_topics['Topic'] == topic, 'Average Sentiment Score'].values[0] + 0.05,
                  y1=top_negative_topics.loc[top_negative_topics['Topic'] == topic, 'Log Proportion of Tokens'].values[0] + 0.05,
                  line=dict(color='red', width=2))

fig.update_traces(marker=dict(line=dict(width=1, color='Black')))
fig.update_layout(showlegend=False)

fig.show()

fig.write_html("sentiment_distribution.html")

# Flask and Dash App
server = Flask(__name__)
app is dash.D