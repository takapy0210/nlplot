import pandas as pd
import pytest

from nlplot import NLPlot


@pytest.fixture
def prepare_data():
    target_col = "text"
    texts = ["Think rich look poor",
             "When you come to a roadblock, take a detour",
             "When it is dark enough, you can see the stars",
             "Never let your memories be greater than your dreams",
             "Victory is sweetest when you’ve known defeat"]
    return pd.DataFrame({target_col: texts})


@pytest.fixture
def prepare_instance(prepare_data):
    npt = NLPlot(prepare_data, taget_col="text")
    return npt


def test_nlplot_bar_ngram(prepare_instance):
    npt = prepare_instance
    npt.bar_ngram(title='uni-gram', ngram=1, top_n=50)


def test_nlplot_treemap(prepare_instance):
    npt = prepare_instance
    npt.treemap(title='Tree of Most Common Words', ngram=1, top_n=30)


def test_nlplot_word_distribution(prepare_instance):
    npt = prepare_instance
    npt.word_distribution(title='number of words distribution')


def test_nlplot_wordcloud(prepare_instance):
    npt = prepare_instance
    npt.wordcloud()


def test_nlplot_co_network(prepare_instance):
    npt = prepare_instance
    npt.build_graph(min_edge_frequency=0)
    npt.co_network(title='Co-occurrence network')


def test_nlplot_sunburst(prepare_instance):
    npt = prepare_instance
    npt.build_graph(min_edge_frequency=0)
    npt.sunburst(title='sunburst chart', colorscale=True)


def test_nlplot_ldavis(prepare_instance):
    npt = prepare_instance
    npt.ldavis(num_topics=5, passes=5, save=False)
