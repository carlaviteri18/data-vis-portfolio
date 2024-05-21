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
import dash
from flask import Flask
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Read the file
filename = 'jane_eyre.txt'
try:
    with open(filename, 'r', encoding='utf8') as f:
        text = f.read()
except:
    with open(filename, 'r', encoding='ISO-8859-1') as f:
        text = f.read()

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
    tokens = [word for word in tokens if not word in stop_words]
    return tokens

# Split the text into chapters
chapters = text.split('CHAPTER ')[1:-1]

# Tokenize chapters
tokenized_chapters = [preprocess(chapter) for chapter in chapters]

# Build bigrams and trigrams
bigram = Phrases(tokenized_chapters, min_count=5, threshold=100)
trigram = Phrases(bigram[tokenized_chapters], threshold=100)

tokenized_chapters = [trigram[bigram[chapter]] for chapter in tokenized_chapters]

# Create dictionary and corpus
dictionary = Dictionary(tokenized_chapters)
dictionary.filter_extremes(no_below=15, no_above=0.5)
corpus = [dictionary.doc2bow(chapter) for chapter in tokenized_chapters]

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Calculate sentiments for each chapter
chapter_sentiments = [sia.polarity_scores(' '.join(chapter))['compound'] for chapter in tokenized_chapters]

# Label chapters based on sentiment
labels = ['positive' if sentiment >= 0.5 else 'negative' if sentiment <= -0.5 else 'neutral' for sentiment in chapter_sentiments]

# Combine chapters with their labels
chapters_with_labels = list(zip(tokenized_chapters, labels))

### Step 2: Tune LDA Model

# Find the optimal number of topics using coherence score
from gensim.models.coherencemodel import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=tokenized_chapters, start=2, limit=40, step=2)

# Plot coherence scores
import matplotlib.pyplot as plt

x = range(2, 40, 2)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Select the model with the highest coherence score
optimal_model = model_list[coherence_values.index(max(coherence_values))]
num_topics = optimal_model.num_topics

# Display the topics
for idx, topic in optimal_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

### Step 3: Enhanced Visualizations

# Prepare data for pyLDAvis
vis = genvis.prepare(optimal_model, corpus, dictionary)

# Save the visualization as an HTML file
pyLDAvis.save_html(vis, 'lda_visualization.html')

### Finding Top 5 Positive and Negative Topics

# Calculate average sentiment scores for each topic
topic_sentiments = np.zeros(num_topics)
topic_token_counts = np.zeros(num_topics)
topic_chapter_counts = np.zeros(num_topics)

for i, (chapter, label) in enumerate(chapters_with_labels):
    topic_distribution = optimal_model.get_document_topics(corpus[i], minimum_probability=0)
    for topic_id, probability in topic_distribution:
        if probability > 0:
            sentiment = sia.polarity_scores(' '.join(chapter))['compound']
            topic_sentiments[topic_id] += sentiment * probability
            topic_token_counts[topic_id] += len(chapter) * probability
            topic_chapter_counts[topic_id] += probability

# Normalize the sentiment scores
average_topic_sentiments = topic_sentiments / np.where(topic_chapter_counts == 0, 1, topic_chapter_counts)

# Prepare the data for Plotly
data = {
    'Topic': np.arange(num_topics),
    'Average Sentiment Score': average_topic_sentiments,
    'Log Proportion of Tokens': np.log1p(topic_token_counts / topic_token_counts.sum()),
    'Token Proportion': topic_token_counts / topic_token_counts.sum(),
    'Number of Chapters': topic_chapter_counts
}

df = pd.DataFrame(data)

# Filter out topics with zero counts in all metrics
df = df[(df['Number of Chapters'] > 0) | (df['Token Proportion'] > 0) | (df['Average Sentiment Score'] != 0)]

# Define color based on sentiment
def get_color(score):
    if score >= 0.5:
        return 'blue'
    elif score <= -0.5:
        return 'red'
    else:
        return 'lightgrey'

df['Color'] = df['Average Sentiment Score'].apply(get_color)

# Find the top 5 positive and negative topics
top_positive_topics = df.nlargest(5, 'Average Sentiment Score')
top_negative_topics = df.nsmallest(5, 'Average Sentiment Score')

print("\nTop 5 Positive Topics:")
print(top_positive_topics[['Topic', 'Average Sentiment Score', 'Token Proportion', 'Number of Chapters']])

print("\nTop 5 Negative Topics:")
print(top_negative_topics[['Topic', 'Average Sentiment Score', 'Token Proportion', 'Number of Chapters']])

### Interactive Visualizations with Plotly

# Create the interactive scatter plot with hover tooltips using Plotly
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

# Highlight top positive and negative topics
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

# Customize the color scale for better visualization
fig.update_traces(marker=dict(line=dict(width=1, color='Black')))
fig.update_layout(showlegend=False)

# Show the plot
fig.show()

# Save the interactive plot as an HTML file
fig.write_html("sentiment_distribution.html")

### Topic Evolution Over Chapters

# Prepare data for topic distribution over chapters
topic_distributions = []

for i, (chapter, label) in enumerate(chapters_with_labels):
    topic_distribution = optimal_model.get_document_topics(corpus[i], minimum_probability=0)
    chapter_distribution = {'Chapter': i+1}
    for topic_id, probability in topic_distribution:
        if probability > 0:
            chapter_distribution[f'Topic {topic_id}'] = probability
    topic_distributions.append(chapter_distribution)

df_topic_distributions = pd.DataFrame(topic_distributions).fillna(0)

# Melt the DataFrame for better plotting with Plotly
df_melted = df_topic_distributions.melt(id_vars=['Chapter'], var_name='Topic', value_name='Probability')

# Filter out topics with zero probability in all chapters
df_melted = df_melted[df_melted['Probability'] > 0]

# Create the interactive visualization
fig = px.line(
    df_melted,
    x='Chapter',
    y='Probability',
    color='Topic',
    line_group='Topic',
    hover_name='Topic',
    hover_data={
        'Probability': ':.2f',
        'Chapter': True
    },
    title='Topic Evolution Over Chapters in Jane Eyre'
)

# Print chapters associated with each topic and their probabilities
for topic in df_melted['Topic'].unique():
    print(f"\nTopic {topic}:")
    topic_chapters = df_melted[df_melted['Topic'] == topic]
    for index, row in topic_chapters.iterrows():
        print(f"Chapter {row['Chapter']}: Probability = {row['Probability']:.4f}")

# Customize the layout
fig.update_layout(
    xaxis_title='Chapter',
    yaxis_title='Topic Probability',
    legend_title='Topics',
    hovermode='x unified'
)

# Show the plot
fig.show()

# Save the interactive plot as an HTML file
fig.write_html("topic_evolution.html")

# Flask and Dash App
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Select three passages from the text
passage_indices = [0, 18, 36]  # Change these indices as needed
passages = [chapters[i] for i in passage_indices]

# Preprocess the passages
preprocessed_passages = [preprocess(p) for p in passages]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3(f"Passage {i+1}"),
            html.Div(id=f'passage-{i+1}', style={'white-space': 'pre-wrap', 'border': '1px solid #ddd', 'padding': '10px'})
        ]) for i in range(3)
    ]),
    dbc.Row([
        dbc.Col([
            html.H5("Select Topic"),
            dcc.Dropdown(
                id='topic-dropdown',
                options=[{'label': f'Topic {i}', 'value': i} for i in range(num_topics)],
                value=0
            )
        ])
    ])
])

# Callback to update the highlighted text
@app.callback(
    [Output(f'passage-{i+1}', 'children') for i in range(3)],
    [Input('topic-dropdown', 'value')]
)
def update_passages(topic_id):
    highlights = []
    for passage, preprocessed_passage in zip(passages, preprocessed_passages):
        highlighted_text = []
        words = passage.split()
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z\s]', '', word).lower()
            prob = dict(optimal_model.get_topic_terms(topic_id, len(dictionary))).get(dictionary.token2id.get(clean_word, -1), 0)
            if prob > 0.000001:
                highlighted_text.append(html.Span(word, style={'background-color': f'rgba(255, 0, 0, {prob*10})'}))
                highlighted_text.append(' ')
            else:
                highlighted_text.append(html.Span(word))
                highlighted_text.append(' ')
        highlights.append(highlighted_text)
    return highlights

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)