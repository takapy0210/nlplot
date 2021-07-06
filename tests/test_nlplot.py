import pandas as pd
import pytest

from nlplot import NLPlot, get_colorpalette, freq_df


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
    npt = NLPlot(prepare_data, target_col="text")
    return npt


@pytest.fixture
def apply_space(prepare_data, target_col="text"):
    df = prepare_data
    return df[target_col].apply(lambda x: ' '.join(x))


@pytest.mark.parametrize("color_palette,"
                         " n_legends",
                         [pytest.param("hls", 0, marks=pytest.mark.xfail),
                          ("hls", 1),
                          ("hls", 10),
                          ("hls", 100)])
def test_get_colorpalette(color_palette, n_legends):
    rgbs = get_colorpalette(color_palette, n_legends)
    assert isinstance(rgbs, list)
    assert len(rgbs) != 0


@pytest.mark.parametrize("n_gram", [pytest.param(0, marks=pytest.mark.xfail),
                                    1, 2, 3])
@pytest.mark.parametrize("top_n", [pytest.param(0, marks=pytest.mark.xfail),
                                   1, 10, 50])
def test_freq_df(apply_space, n_gram, top_n):
    stop_words = []
    verbose = False
    df = apply_space
    word_frequency = freq_df(df, n_gram=n_gram, n=top_n,
                             stopwords=stop_words, verbose=verbose)
    expect_columns = ["word", "word_count"]
    assert isinstance(word_frequency, pd.DataFrame)
    assert word_frequency.ndim == 2
    assert len(word_frequency) <= top_n
    assert list(word_frequency.columns) == expect_columns


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
