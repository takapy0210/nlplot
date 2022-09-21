# 📝 nlplot
nlplot: Analysis and visualization module for Natural Language Processing 📈

## Description
Facilitates the visualization of natural language processing and provides quicker analysis

You can draw the following graph

1. [N-gram bar chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_uni-gram.html)
2. [N-gram tree Map](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Tree%20of%20Most%20Common%20Words.html)
3. [Histogram of the word count](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_number%20of%20words%20distribution.html)
4. [wordcloud](https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/wordcloud.png)
5. [co-occurrence networks](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Co-occurrence%20network.html)
6. [sunburst chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_sunburst%20chart.html)

（Tested in English and Japanese）

## Requirement
- [python package](https://github.com/takapy0210/nlplot/blob/master/requirements.txt)

## Installation
```sh
pip install nlplot
```

I've posted on [this blog](https://www.takapy.work/entry/2020/05/17/192947) about the specific use. (Japanese)

And, The sample code is also available [in the kernel of kaggle](https://www.kaggle.com/takanobu0210/twitter-sentiment-eda-using-nlplot). (English)

## Quick start - Data Preparation

The column to be analyzed must be a space-delimited string

```python
# sample data
target_col = "text"
texts = [
    "Think rich look poor",
    "When you come to a roadblock, take a detour",
    "When it is dark enough, you can see the stars",
    "Never let your memories be greater than your dreams",
    "Victory is sweetest when you’ve known defeat"
    ]
df = pd.DataFrame({target_col: texts})
df.head()
```

|    |  text  |
| ---- | ---- |
|  0  |  Think rich look poor |
|  1  |  When you come to a roadblock, take a detour |
|  2  |  When it is dark enough, you can see the stars |
|  3  |  Never let your memories be greater than your dreams  |
|  4  |  Victory is sweetest when you’ve known defeat  |


## Quick start - Python API
```python
import nlplot
import pandas as pd
import plotly
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt

%matplotlib inline

# target_col as a list type or a string separated by a space.
npt = nlplot.NLPlot(df, target_col='text')

# Stopword calculations can be performed.
stopwords = npt.get_stopword(top_n=30, min_freq=0)

# 1. N-gram bar chart
fig_unigram = npt.bar_ngram(
    title='uni-gram',
    xaxis_label='word_count',
    yaxis_label='word',
    ngram=1,
    top_n=50,
    width=800,
    height=1100,
    color=None,
    horizon=True,
    stopwords=stopwords,
    verbose=False,
    save=False,
)
fig_unigram.show()

fig_bigram = npt.bar_ngram(
    title='bi-gram',
    xaxis_label='word_count',
    yaxis_label='word',
    ngram=2,
    top_n=50,
    width=800,
    height=1100,
    color=None,
    horizon=True,
    stopwords=stopwords,
    verbose=False,
    save=False,
)
fig_bigram.show()


# 2. N-gram tree Map
fig_treemap = npt.treemap(
    title='Tree map',
    ngram=1,
    top_n=50,
    width=1300,
    height=600,
    stopwords=stopwords,
    verbose=False,
    save=False
)
fig_treemap.show()


# 3. Histogram of the word count
fig_histgram = npt.word_distribution(
    title='word distribution',
    xaxis_label='count',
    yaxis_label='',
    width=1000,
    height=500,
    color=None,
    template='plotly',
    bins=None,
    save=False,
)
fig_histgram.show()


# 4. wordcloud
fig_wc = npt.wordcloud(
    width=1000,
    height=600,
    max_words=100,
    max_font_size=100,
    colormap='tab20_r',
    stopwords=stopwords,
    mask_file=None,
    save=False
)
plt.figure(figsize=(15, 25))
plt.imshow(fig_wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# 5. co-occurrence networks
npt.build_graph(stopwords=stopwords, min_edge_frequency=10)
# The number of nodes and edges to which this output is plotted.
# If this number is too large, plotting will take a long time, so adjust the [min_edge_frequency] well.
# >> node_size:70, edge_size:166
fig_co_network = npt.co_network(
    title='Co-occurrence network',
    sizing=100,
    node_size='adjacency_frequency',
    color_palette='hls',
    width=1100,
    height=700,
    save=False
)
iplot(fig_co_network)


# 6. sunburst chart
fig_sunburst = npt.sunburst(
    title='sunburst chart',
    colorscale=True,
    color_continuous_scale='Oryel',
    width=1000,
    height=800,
    save=False
)
fig_sunburst.show()


# other
# The original data frame of the co-occurrence network can also be accessed
display(
    npt.node_df.head(), npt.node_df.shape,
    npt.edge_df.head(), npt.edge_df.shape
)

```

## Document
TBD

## Test
```sh
cd tests
pytest
```

## Other

- Plotly is used to plot the figure
    - https://plotly.com/python/

- co-occurrence networks is used to calculate the co-occurrence network
    - https://networkx.github.io/documentation/stable/tutorial.html

- wordcloud uses the following fonts
    - https://mplus-fonts.osdn.jp/about.html
