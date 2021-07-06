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

# target_col as a list type or a string separated by a space.
npt = nlplot.NLPlot(df, target_col='text')

# Stopword calculations can be performed.
stopwords = npt.get_stopword(top_n=30, min_freq=0)

# 1. N-gram bar chart
npt.bar_ngram(title='uni-gram', ngram=1, top_n=50, stopwords=stopwords)
npt.bar_ngram(title='bi-gram', ngram=2, top_n=50, stopwords=stopwords)

# 2. N-gram tree Map
npt.treemap(title='Tree of Most Common Words', ngram=1, top_n=30, stopwords=stopwords)

# 3. Histogram of the word count
npt.word_distribution(title='words distribution')

# 4. wordcloud
npt.wordcloud(stopwords=stopwords, colormap='tab20_r')

# 5. co-occurrence networks
npt.build_graph(stopwords=stopwords, min_edge_frequency=10)
# The number of nodes and edges to which this output is plotted.
# If this number is too large, plotting will take a long time, so adjust the [min_edge_frequency] well.
>> node_size:70, edge_size:166
npt.co_network(title='Co-occurrence network')

# 6. sunburst chart
npt.sunburst(title='sunburst chart', colorscale=True)

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
