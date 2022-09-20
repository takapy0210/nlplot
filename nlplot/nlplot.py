"""Visualization Module for Natural Language Processing"""

import os
import gc
import itertools
from io import BytesIO
from PIL import Image
from collections import defaultdict, Counter
import datetime as datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx
from networkx.algorithms import community

# Load Japanese Font
TTF_FILE_NAME = str(os.path.dirname(__file__)) + '/data/mplus-1c-regular.ttf'


def get_colorpalette(colorpalette: str, n_colors: int) -> list:
    """Get a color palette

    Args:
        colorpalette (str): cf.https://qiita.com/SaitoTsutomu/items/c79c9973a92e1e2c77a7 .
        n_colors (int): Number of colors to be displayed.

    Returns:
        list of str: e.g. ['rgb(220.16,95.0272,87.03)', 'rgb(220.16,209.13005714285714,87.03)', ...]
    """
    palette = sns.color_palette(colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb]) for rgb in palette]
    return rgb


def generate_freq_df(value: pd.Series, n_gram: int = 1, top_n: int = 50, stopwords: list = [],
                     verbose: bool = True) -> pd.DataFrame:
    """Generate a data frame of frequent word

    Args:
        value (pd.Series): Separated by space values.
        n_gram (int, optional): N number of N grams. Dafaults to 1.
        top_n (int, optional): N to get TOP N. Dafaults to 50.
        stopwords (list of str, optional): A list of words to specify for the stopword.
        verbose (bool, optional): Whether or not to output the log by tqdm.

    Returns:
        pd.DataFrame: frequent word DataFrame
    """

    def generate_ngrams(text: str, n_gram: int = 1) -> list:
        """Generate ngram

        Args:
            text (str): Target text
            n_gram (int, optional): N number of N grams. Defaults to 1.

        Returns:
            list of str: ngram list
        """
        token = [token for token in text.lower().split(" ")
                 if token != "" if token not in stopwords]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    freq_dict = defaultdict(int)
    if verbose:
        for sent in tqdm(value):
            for word in generate_ngrams(str(sent), n_gram):
                freq_dict[word] += 1
    else:
        for sent in value:
            for word in generate_ngrams(str(sent), n_gram):
                freq_dict[word] += 1

    # frequent word DataFrame
    output_df = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    output_df.columns = ['word', 'word_count']
    return output_df.head(top_n)


class NLPlot():
    """Visualization Module for Natural Language Processing

    Attributes:
        df (pd.DataFrame): Original dataframe to be graphed.
        target_col: Columns to be analyzed that exist in df (assuming type list) e.g. [hoge, fuga, ...]
        output_file_path: path to save the html file of the generated graph
        default_stopwords_file_path: The path to the file that defines the default stopword
    """

    def __init__(self, df: pd.DataFrame, target_col: str,
                 output_file_path: str = './', default_stopwords_file_path: str = ''):
        self.df = df.copy()
        self.target_col = target_col
        self.df.dropna(subset=[self.target_col], inplace=True)
        if type(self.df[self.target_col].iloc[0]) is not list:
            self.df.loc[:, self.target_col] = self.df[self.target_col].map(lambda x: x.split())
        self.output_file_path = output_file_path
        self.default_stopwords = []
        if os.path.exists(default_stopwords_file_path):
            f = open(default_stopwords_file_path)
            txt_file = f.readlines()
            f.close()
            self.default_stopwords = [line.strip() for line in txt_file]

    def get_stopword(self, top_n: int = 10, min_freq: int = 5) -> list:
        """Calculate the stop word.

        Calculate the top_n words with the highest number of occurrences
        and the words that occur only below the min_freq as stopwords.

        Args:
            top_n (int, optional): Top N of the number of occurrences of words to exclude. Defaults to 10.
            min_freq (int, optional): Bottom of the number of occurrences of words to exclude. Defaults to 5.

        Returns:
            list: list of stop words
        """
        fdist = Counter()

        # Count the number of occurrences per word.
        for doc in self.df[self.target_col]:
            for word in doc:
                fdist[word] += 1
        # word with a high frequency
        common_words = {word for word, freq in fdist.most_common(top_n)}
        # word with a low frequency
        rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
        stopwords = list(common_words.union(rare_words))
        return stopwords

    def bar_ngram(
        self,
        title: str = None,
        xaxis_label: str = '',
        yaxis_label: str = '',
        ngram: int = 1,
        top_n: int = 50,
        width: int = 800,
        height: int = 1100,
        color: str = None,
        horizon: bool = True,
        stopwords: list = [],
        verbose: bool = True,
        save: bool = False
    ) -> px.bar:
        """Plots of n-gram bar chart

        Args:
            title (str, optional): title of plot. Defaults to None.
            xaxis_label (str, optional): x-axis label name. Defaults to ''.
            yaxis_label (str, optional): y-axis label name. Defaults to ''.
            n_gram (int, optional): N number of N grams. Defaults to 1.
            top_n (int, optional): How many words should be output. Defaults to 50.
            width (int, optional): width of the graph. Defaults to 800.
            height (int, optional): height of the graph. Defaults to 1100.
            color (str, optional): color of the chart. Defaults to None.
            horizon (bool, optional): To create a horizontal bar graph, use the True. Defaults to True.
            stopwords (list, optional): A list of words to specify for the stopword. Defaults to [].
            verbose (bool, optional): Whether or not to output the log by tqdm. Defaults to True.
            save (bool, optional): Whether or not to save the HTML file. Defaults to False.

        Returns:
            px.bar: Figure of a bar graph
        """

        stopwords += self.default_stopwords

        self.ngram_df = self.df.copy()
        self.ngram_df.loc[:, 'space'] = self.ngram_df[self.target_col].apply(lambda x: ' '.join(x))

        # word count
        self.ngram_df = generate_freq_df(self.ngram_df['space'], n_gram=ngram, top_n=top_n,
                                         stopwords=stopwords, verbose=verbose)

        if horizon:
            fig = px.bar(
                self.ngram_df.sort_values('word_count'),
                y='word',
                x='word_count',
                text='word_count',
                orientation='h',)
        else:
            fig = px.bar(
                self.ngram_df,
                y='word_count',
                x='word',
                text='word_count',)

        fig.update_traces(
            texttemplate='%{text:.2s}',
            textposition='auto',
            marker_color=color,)
        fig.update_layout(
            title=str(title),
            xaxis_title=str(xaxis_label),
            yaxis_title=str(yaxis_label),
            width=width,
            height=height,)

        if save:
            self.save_plot(fig, title)

        return fig

    def treemap(
        self,
        title: str = None,
        ngram: int = 1,
        top_n: int = 50,
        width: int = 1300,
        height: int = 600,
        stopwords: list = [],
        verbose: bool = True,
        save: bool = False
    ) -> px.treemap:
        """Plots of Tree Map

        Args:
            title (str, optional): title of plot. Defaults to None.
            ngram (int, optional): N number of N grams. Defaults to 1.
            top_n (int, optional): How many words should be output. Defaults to 50.
            width (int, optional): width of the graph. Defaults to 1300.
            height (int, optional): height of the graph. Defaults to 600.
            stopwords (list os str, optional): A list of words to specify for the stopword. Defaults to [].
            verbose (bool, optional): Whether or not to output the log by tqdm. Defaults to True.
            save (bool, optional): Whether or not to save the HTML file. Defaults to False.

        Returns:
            px.treemap: Figure of a treemap graph
        """

        stopwords += self.default_stopwords

        self.treemap_df = self.df.copy()
        self.treemap_df.loc[:, 'space'] = self.treemap_df[self.target_col].apply(lambda x: ' '.join(x))

        # word count
        self.treemap_df = generate_freq_df(self.treemap_df['space'], n_gram=ngram, top_n=top_n,
                                           stopwords=stopwords, verbose=verbose)

        fig = px.treemap(
            self.treemap_df,
            path=['word'],
            values='word_count',
        )
        fig.update_layout(
            title=str(title),
            width=width,
            height=height,
        )

        if save:
            self.save_plot(fig, title)

        return fig

    def word_distribution(
        self,
        title: str = None,
        xaxis_label: str = '',
        yaxis_label: str = '',
        width: int = 1000,
        height: int = 600,
        color: str = None,
        template: str = 'plotly',
        bins: int = None,
        save: bool = False
    ) -> px.histogram:
        """Plots of word count histogram

        Args:
            title (str, optional): title of plot. Defaults to None.
            xaxis_label (str, optional): x-axis label name. Defaults to ''.
            yaxis_label (str, optional): y-axis label name. Defaults to ''.
            width (int, optional): width of the graph. Defaults to 100.
            height (int, optional): height of the graph. Defaults to 600.
            color (str, optional): color of the chart. Defaults to None.
            template (str, optional): The plotly drawing style. Defaults to 'plotly'.
            bins (int, optional): Number of bins. Defaults to None.
            save (bool, optional): Whether or not to save the HTML file. Defaults to False.

        Returns:
            px.histogram: Figure of a bar graph
        """
        self.df.loc[:, self.target_col + '_length'] = self.df[self.target_col].map(lambda x: len(x))
        fig = px.histogram(self.df, x=self.target_col+'_length', color=color, template=template, nbins=bins)
        fig.update_layout(
            title=str(title),
            xaxis_title=str(xaxis_label),
            yaxis_title=str(yaxis_label),
            width=width,
            height=height,)

        if save:
            self.save_plot(fig, title)

        return fig

    def wordcloud(
        self,
        width: int = 800,
        height: int = 500,
        max_words: int = 100,
        max_font_size: int = 80,
        stopwords: list = [],
        colormap: str = None,
        mask_file: str = None,
        save: bool = False
    ) -> WordCloud:
        """Plots of WordCloud

        Args:
            width (int, optional): width of the graph. Defaults to 800.
            height (int, optional): height of the graph. Defaults to 500.
            max_words (int, optional): Number of words to display. Defaults to 100.
            max_font_size (int, optional): Size of words to be displayed. Defaults to 80.
            stopwords (list, optional): A list of words to specify for the stopword.. Defaults to [].
            colormap (str, optional): cf.https://karupoimou.hatenablog.com/entry/2019/05/17/153207. Defaults to None.
            mask_file (str, optional): Image to be masked file. Defaults to None.
            save (bool, optional): Whether or not to save the Image file. Defaults to False.

        Returns:
            WordCloud
        """

        f_path = TTF_FILE_NAME
        if mask_file is not None:
            mask = np.array(Image.open(mask_file))
        else:
            mask = None

        text = self.df[self.target_col]
        stopwords += self.default_stopwords

        wordcloud = WordCloud(
            background_color='white',
            font_step=1,
            contour_width=0,
            contour_color='steelblue',
            font_path=f_path,
            stopwords=stopwords,
            max_words=max_words,
            max_font_size=max_font_size,
            random_state=42,
            width=width,
            height=height,
            mask=mask,
            collocations=False,
            prefer_horizontal=1,
            colormap=colormap)
        wordcloud = wordcloud.generate(' '.join(list(itertools.chain(*list(text)))))

        if save:
            def save_array(img):
                stream = BytesIO()
                if save:
                    date = str(pd.to_datetime(datetime.datetime.now())).split(' ')[0]
                    filename = date + '_wordcloud.png'
                    Image.fromarray(img).save(self.output_file_path + filename)
                Image.fromarray(img).save(stream, 'png')

            img = wordcloud.to_array()
            save_array(img)

        return wordcloud

    def get_edges_nodes(self, batches, min_edge_frequency) -> None:
        """Generating the Edge and Node data frames for a graph

        Args:
            batches (list): array of word lists
            min_edge_frequency (int): Minimum number of edge occurrences.
                                      Edges less than this number will be removed.

        Returns:
            None
        """

        def _ranked_topics(batches):
            """sort function"""
            batches.sort()
            return batches

        def _unique_combinations(batches):
            """craeted unique combinations

                e.g. [('hoge1', 'hoge2'), ('hoge1', 'hoge3'), ...]

            """
            return list(itertools.combinations(_ranked_topics(batches), 2))

        def _add_unique_combinations(_unique_combinations, _dict):
            """Calculate how many times the combination appears and store it in a dictionary"""
            for combination in _unique_combinations:
                if combination in _dict:
                    _dict[combination] += 1
                else:
                    _dict[combination] = 1
            return _dict

        edge_dict = {}
        source = []
        target = []
        edge_frequency = []
        for batch in batches:
            # e.g. {('hoge1', 'hoge2'): 8, ('hoge1', 'hoge3'): 3, ...}
            edge_dict = _add_unique_combinations(_unique_combinations(batch), edge_dict)

        # create edge dataframe
        for key, value in edge_dict.items():
            source.append(key[0])
            target.append(key[1])
            edge_frequency.append(value)
        edge_df = pd.DataFrame({'source': source, 'target': target, 'edge_frequency': edge_frequency})
        edge_df.sort_values(by='edge_frequency', ascending=False, inplace=True)
        edge_df.reset_index(drop=True, inplace=True)
        edge_df = edge_df[edge_df['edge_frequency'] > min_edge_frequency]

        # create node dataframe
        node_df = pd.DataFrame({'id': list(set(list(edge_df['source']) + list(edge_df['target'])))})
        labels = [i for i in range(len(node_df['id']))]
        node_df.loc[:, 'id_code'] = node_df.index
        node_dict = dict(zip(node_df['id'], labels))

        edge_df.loc[:, 'source_code'] = edge_df['source'].apply(lambda x: node_dict[x])
        edge_df.loc[:, 'target_code'] = edge_df['target'].apply(lambda x: node_dict[x])

        self.edge_df = edge_df
        self.node_df = node_df
        self.node_dict = node_dict
        self.edge_dict = edge_dict

        return None

    def get_graph(self) -> nx.Graph:
        """create Networkx

        Returns:
            nx.Graph(): Networkx graph
        """

        def _extract_edges(edge_df):
            tuple_out = []
            for i in range(0, len(self.edge_df.index)):
                tuple_out.append((self.edge_df['source_code'][i], self.edge_df['target_code'][i]))
            return tuple_out

        # Networkx graph
        G = nx.Graph()

        # Add Node from Data Frame
        G.add_nodes_from(self.node_df.id_code)

        # Add Edge from Data Frame
        # e.g. [(8, 47), (4, 47), (47, 0), ...]
        edge_tuples = _extract_edges(self.edge_df)
        for i in edge_tuples:
            G.add_edge(i[0], i[1])

        return G

    def build_graph(self, stopwords=[], min_edge_frequency=10) -> None:
        """Preprocessing to output a co-occurrence network

        Args:
            stopwords (list): List of words to exclude
            min_edge_frequency (int): Minimum number of edge occurrences
                                      (edges with fewer than this number are excluded)

        Returns:
            None
        """

        self.df_edit = self.df.copy()

        # Remove duplicates from the list to be analyzed
        self.df_edit.loc[:, self.target_col] = self.df_edit[self.target_col].map(lambda x: list(set(x)))

        # Acquire only the column data for this analysis.
        self.target = self.df_edit[[self.target_col]]

        # Get an array of word lists by excluding stop words
        # [['hoge1', 'hoge4', 'hoge7', 'hoge5'],
        #  ['hoge7', 'hoge2', 'hoge9', 'hoge12', 'hoge4'],...]
        stopwords += self.default_stopwords

        def _removestop(words):
            for stop_word in stopwords:
                try:
                    words.remove(stop_word)
                    words = words
                except Exception:
                    pass
            return words
        batch = self.target[self.target_col].map(_removestop)
        batches = batch.values.tolist()

        # Generating the Edge and Node data frames for a graph
        self.get_edges_nodes(batches, min_edge_frequency)

        # create adjacency, centrality, cluster
        # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.adjacency.html?highlight=adjacency#networkx.Graph.adjacency
        # https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html#betweenness-centrality
        # https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.cluster.clustering.html?highlight=clustering#clustering
        self.G = self.get_graph()
        self.adjacencies = dict(self.G.adjacency())
        self.betweeness = nx.betweenness_centrality(self.G)
        self.clustering_coeff = nx.clustering(self.G)
        self.node_df.loc[:, 'adjacency_frequency'] = self.node_df['id_code'].map(lambda x: len(self.adjacencies[x]))
        self.node_df.loc[:, 'betweeness_centrality'] = self.node_df['id_code'].map(lambda x: self.betweeness[x])
        self.node_df.loc[:, 'clustering_coefficient'] = self.node_df['id_code'].map(lambda x: self.clustering_coeff[x])

        # create community
        # https://networkx.github.io/documentation/stable/reference/algorithms/community.html#module-networkx.algorithms.community.modularity_max
        self.communities = community.greedy_modularity_communities(self.G)
        self.communities_dict = {}
        nodes_in_community = [list(i) for i in self.communities]
        for i in nodes_in_community:
            self.communities_dict[nodes_in_community.index(i)] = i

        def community_allocation(source_val):
            for k, v in self.communities_dict.items():
                if source_val in v:
                    return k

        self.node_df.loc[:, 'community'] = self.node_df['id_code'].map(lambda x: community_allocation(x))

        print('node_size:{}, edge_size:{}'.format(self.node_df.shape[0], self.edge_df.shape[0]))

        return None

    def co_network(self, title, sizing=100, node_size='adjacency_frequency',
                   color_palette='hls', layout=nx.kamada_kawai_layout,
                   light_theme=True, width=1700, height=1200, save=False) -> go:
        """Plots of co-occurrence networks

        Args:
            title (str): title of plot
            sizing (int): Size of the maker
            node_size (str): Column name to specify the size of the node
            color_palette (str): cf.https://qiita.com/SaitoTsutomu/items/c79c9973a92e1e2c77a7
            layout : nx.kamada_kawai_layout
            light_theme (bool): True if you want a light theme.
            width (int): width of the graph
            height (int): height of the graph
            save (bool): Whether or not to save the HTML file.

        Returns:
            plotly.graph_objs
        """

        # formatting options for plot - dark vs. light theme
        if light_theme:
            back_col = '#ffffff'
            edge_col = '#ece8e8'
        else:
            back_col = '#000000'
            edge_col = '#2d2b2b'

        # select of node_df -> ['adjacency_frequency', 'betweeness_centrality', 'clustering_coefficient']
        X = self.node_df[self.node_df.columns[2:5]]
        cols = self.node_df.columns[2:5]

        # scaling
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        _df = pd.DataFrame(X_scaled)
        _df.columns = cols

        for i in _df.columns:
            _df.loc[:, i] = _df[i].apply(lambda x: x*sizing)

        # extract graph x,y co-ordinates from G instance
        pos = layout(self.G)

        # add position of each node from G to 'pos' key
        for node in self.G.nodes:
            self.G.nodes[node]['pos'] = list(pos[node])

        stack = []
        index = 0

        # add edges to Plotly go.Scatter object
        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]['pos']
            x1, y1 = self.G.nodes[edge[1]]['pos']
            weight = 1.2
            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                               mode='lines',
                               line={'width': weight},
                               marker=dict(color=edge_col),
                               line_shape='spline',
                               opacity=1)

            # append edge traces
            stack.append(trace)

            index = index + 1

        # make a partly empty dictionary for the nodes
        marker = {'size': [], 'line': dict(width=0.5, color=edge_col), 'color': []}

        # initialise a go.Scatter object for the nodes
        node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[],
                                mode='markers+text', textposition='bottom center',
                                hoverinfo="text", marker=marker)

        index = 0

        n_legends = len(self.node_df['community'].unique())
        colors = get_colorpalette(color_palette, n_legends)

        # add nodes to Plotly go.Scatter object
        for node in self.G.nodes():

            x, y = self.G.nodes[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([self.node_df['id'][index]])

            # Change the color scheme for each community
            for i in range(n_legends):
                if self.node_df.community[index] == i:
                    node_trace['marker']['color'] += tuple([list(colors)[i]])
            node_trace['marker']['size'] += tuple([list(_df[node_size])[index]])

            index = index + 1

        # append node traces
        stack.append(node_trace)

        # set up axis for plot
        # hide axis line, grid, ticklabels and title
        axis = dict(showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title='')

        # set up figure for plot
        fig = {
            "data": stack,
            "layout":
                go.Layout(title=str(title),
                          font=dict(family='Arial', size=12),
                          width=width,
                          height=height,
                          autosize=True,
                          showlegend=False,
                          xaxis=axis,
                          yaxis=axis,
                          margin=dict(l=40, r=40, b=85, t=100, pad=0),
                          hovermode='closest',
                          plot_bgcolor=back_col,  # set background color
                          )
        }

        if save:
            self.save_plot(fig, title)

        del _df
        gc.collect()
        return fig

    def sunburst(self, title, colorscale=False, color_col='betweeness_centrality',
                 color_continuous_scale='Oryel', width=1100, height=1100, save=False) -> px.sunburst:
        """Plots of sunburst chart

        Args:
            title (str): title of plot
            colorscale (bool): Size of the maker
            color_col (str): Color-coded column names.
            color_continuous_scale (str): cf.https://plotly.com/python/builtin-colorscales/
            width (int): width of the graph
            height (int): height of the graph
            save (bool): Whether or not to save the HTML file.

        Returns:
            px.sunburst: Figure of a sunburst graph

        """

        # make copy of node dataframe
        _df = self.node_df.copy()

        # change community label to string (needed for plot)
        _df.loc[:, 'community'] = _df['community'].map(lambda x: str(x))

        # conditionals for plot type
        if colorscale is False:
            fig = px.sunburst(_df, path=['community', 'id'], values='adjacency_frequency',
                              color='community', hover_name=None, hover_data=None)
        else:
            fig = px.sunburst(_df, path=['community', 'id'], values='adjacency_frequency',
                              color=color_col, hover_data=None,
                              color_continuous_scale=color_continuous_scale,
                              color_continuous_midpoint=np.average(_df[color_col],
                              weights=_df[color_col]))

        fig.update_layout(
            title=str(title),
            width=width,
            height=height,)

        if save:
            self.save_plot(fig, title)

        del _df
        gc.collect()
        return fig

    def save_plot(self, fig, title) -> None:
        """Save the HTML file

        Args:
            fig (fig): Figure to be saved
            title (str): File name to save

        Returns:
            None
        """
        date = str(pd.to_datetime(datetime.datetime.now())).split(' ')[0]
        filename = date + '_' + str(title) + '.html'
        filename = self.output_file_path + filename
        plotly.offline.plot(fig, filename=filename, auto_open=False)
        return None

    def save_tables(self) -> None:
        """Storing a data frame"""

        date = str(pd.to_datetime(datetime.datetime.now())).split(' ')[0]
        self.node_df.to_csv(date + "_node_df_" + self.source + ".csv", index=False)
        print('Saved nodes')
        self.edge_df.to_csv(date + "_edge_df_" + self.source + ".csv", index=False)
        print('Saved edges')
        self.df.to_csv(date + "_df_" + self.source + "_.csv", index=False)
        print('Saved unedited dataframe')
        return None
