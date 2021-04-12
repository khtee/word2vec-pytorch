import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.manifold import TSNE

import torch
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk_corpora = ['words', 'stopwords', 'wordnet']

for corpora in nltk_corpora:
    try:
        nltk.data.find('corpora/' + corpora)
    except LookupError:
        nltk.download(corpora)


def save_pickle(output_file, data):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
        f.close()


def load_pickle(input_file):
    with open(input_file, "rb") as f:
        data = pickle.load(f)
        f.close()
    return data


def preprocess_raw_text(source_file, output_file, min_length=5):
    """Preprocess raw text file line by line.

    Raw text are read line by line and performs following preprocessing.
    1. Tokenize using regular expression.
        a. Remove any numeric and punctuation.
        b. Remove words less than 2 characters.
    2. Lemmatize words to their root form.
    3. Remove any non English words.
    4. Remove stop words using NLTK.
    5. Only keep the new sentence if the length is more than min_length.

    Args:
        source_file: A string for input file.
        output_file: A string for output file.
        min_length: An integer for minimum sentence length

    Returns:
        Printout the total sentence read and number of new sentences.
    """
    STOP_WORDS = set(stopwords.words('english'))
    TOKENIZER = RegexpTokenizer(r'[a-zA-Z]{2,}')
    WORDS = set(nltk.corpus.words.words())

    total_sent = 0
    sent_count = 0
    write_file = open(output_file, 'a+')

    with open(source_file, 'r') as read_file:
        for line in read_file:
            total_sent += 1
            line = line.lower()
            text = TOKENIZER.tokenize(line)
            filtered_sentence = [
                w for w in text if w not in STOP_WORDS and w in WORDS
            ]
            if len(filtered_sentence) > min_length:
                write_file.write(" ".join(filtered_sentence) + '\n')
                sent_count += 1

            if total_sent % 1000000 == 0:
                print(str(total_sent // 1000000),
                      ' million sentences processed.')

        read_file.close()
    write_file.close()

    print('total sentence before processed: %i' % (total_sent))
    print('total sentence after processed: %i' % (sent_count))


def get_context(words, position, window_size=2):
    """Get all context words with given window size.

    Args:
        words: A list of numeric for word index.
        position: A int for current position of target word in words.
        window_size: A int for the window size.

    Return:
        A list of context words.
    """
    start = max(0, position - window_size)
    stop = min(position + window_size, len(words))
    context_words = words[start:position] + words[position + 1:stop + 1]

    return list(context_words)


def create_neg_sample_list(word_counts):
    """Create a list of all negative samples.

    The number of element for each word is based on its occurence in corpus.
    Word with higher frequency will have higher probability of being chosen
    as negative sample.

    Args:
        vocab_list: A list of numeric for word occurence.

    Returns:
        A list of all negative samples.
    """
    negatives = []
    pow_freq = np.array(list(word_counts.values()))**0.75
    sum_pow_freq = np.sum(pow_freq)
    ratio = pow_freq / sum_pow_freq
    count = np.round(ratio * 1e6)
    max_sample_id = len(count)
    for wid, c in enumerate(count):
        negatives += [wid] * int(c)
    negatives = np.array(negatives)
    np.random.shuffle(negatives)
    return negatives


def get_negatives(target, negatives, N, K=5):
    """Obtain negative samples

    Pick a random number.
    Those elements that appear more have a higher probability.

    Args:
        target: A list of word embedding.
        negatives: A list of negative samples.
        N: An integer for number of targets
        K: An integer for number of negative samples per target.

    Return:
        A list of lists for negative samples.
    """
    res = np.random.choice(negatives, size=(N, K))
    for i in range(len(res)):
        for j in range(K):
            if res[i, j] == target[i]:
                res[i, j] = np.random.choice(negatives, size=1)

    return res


def weight_func(x, x_max, alpha):
    wx = (x / x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx


def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)


def calc_spearman_coef(input_file, word2idx, embeddings):
    """Calculate the spearman coefficient for wordsim file.

    Either pass a ndarry of embedding or .npy file. If a .npy file is
    specified then it will be priority.

    Args:
        input_file: A string for wordsim CSV file.
        word2idx: A dictionary for word to index mapping.
        embeddings: A numpy array for trained word embeddings.
        embed_file: A npy file storing word embeddings.

    Return:
        Spearman coefficient score.

    """
    wordsim_file = pd.read_csv(input_file)
    wordsim_file = np.array(wordsim_file)

    cos_dict_id = dict()
    wordsim_id = dict()

    for (word_a, word_b, num) in wordsim_file:
        if word_a in word2idx and word_b in word2idx:
            wordsim_id[(word2idx.get(word_a, 0), word2idx.get(word_b,
                                                              0))] = num / 10

    # Compute Cosine Similarities
    for (id_a, id_b), value in wordsim_id.items():

        embeddings_a = embeddings[id_a].reshape(1, -1)
        embeddings_b = embeddings[id_b].reshape(1, -1)

        similarity = np.ndarray.item(
            cosine_similarity(embeddings_a, embeddings_b)[0])
        similarity = round(similarity, 2)

        cos_dict_id[id_a, id_b] = similarity

    a = list([])
    b = list([])
    for (id_a, id_b), value in wordsim_id.items():
        if cos_dict_id.get((id_a, id_b)):
            a.append(value)
            b.append(cos_dict_id[(id_a, id_b)])

    spear = spearmanr(a, b)

    return (spear[0])


def visualize_embedding(embeddings, idx2word, num_words=300, title=""):
    """ Visualize word embedding in 2D space.

    Args:
        embeddings: A ndarray of word embeddings.
        idx2word: A dictionary for index to word mapping.
        num_words: An integer for number of words to be plotted.
        title: A string for the plot title.
    """
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:num_words, :])
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(num_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(idx2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]),
                     alpha=0.7)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    plt.show()


class PreprocessData:
    def __init__(self, wiki, min_count=5, cooc=False):
        self.wiki = wiki
        self.min_count = min_count
        self.TOKENIZER = RegexpTokenizer(r'[a-zA-Z]{2,}')
        self.words = set(nltk.corpus.words.words())
        self.stop_words = set(stopwords.words('english'))
        self.LEMMATIZER = WordNetLemmatizer()
        self.word_counts = Counter()
        self._get_word_counts()
        self.total_words = Counter()
        self._get_total_words()
        self.vocab = list()
        self.word_count_filtered = dict()
        self._text_cleaning()
        self.num_vocab = len(self.vocab)
        self.word2idx = Counter()
        self.idx2word = Counter()
        self._create_lookup_tables()
        self.threshold = 1e-3
        self.drop_prob = dict()
        self._get_drop_prob()

        print("Number of vocabulary: %s" % (self.num_vocab))
        print("Total number of words: %s" % (self.total_words))

    def _get_word_counts(self):
        word_counts = Counter()
        bufsize = 65536
        file1 = open(self.wiki, 'r')
        while True:
            lines = file1.readlines(bufsize)
            if not lines:
                break
            for line in lines:
                line = line.lower()
                text = self.TOKENIZER.tokenize(line)
                text = [self.LEMMATIZER.lemmatize(w) for w in text]
                text = [
                    w for w in text
                    if w in self.words and w not in self.stop_words
                ]
                self.word_counts.update(Counter(text))
        print('done get word counts')

    def _text_cleaning(self):
        for k, v in self.word_counts.items():
            if self.total_words >= self.min_count:
                self.vocab.append(k)
                self.word_count_filtered[k] = v

    def _create_lookup_tables(self):
        self.idx2word = {i: word for i, word in enumerate(self.vocab)}
        self.word2idx = {word: i for i, word in self.idx2word.items()}
        print('done create lookup tables')

    def _get_total_words(self):
        self.total_words = 0
        for count in self.word_counts.values():
            self.total_words += count
        print('done get total words')

    def _get_drop_prob(self):
        freqs = {
            word: count / self.total_words
            for word, count in self.word_counts.items()
        }
        self.drop_prob = {
            word: (np.sqrt(freqs[word] / self.threshold) + 1) *
            (self.threshold / freqs[word])
            for word in self.word_counts
        }
        print('done get drop prob')


class Word2VecIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        file_itr = open(self.file_path)
        return file_itr


class PreprocessDataGlove:
    def __init__(self, text, max_words=10000, window_size=2):
        self.window_size = window_size
        self.TOKENIZER = RegexpTokenizer(r'[a-zA-Z]{2,}')

        data = ""
        word_count = 0
        with open(text, 'r') as f:
            for line in f:
                split_line = self.TOKENIZER.tokenize(line)
                data += " ".join(split_line) + " "
                word_count += len(split_line)

                if word_count > max_words:
                    break
            f.close()

        self.tokens = self.TOKENIZER.tokenize(data)
        self.word_counter = Counter()
        self.word_counter.update(self.tokens)
        self.word2idx = {
            w: i
            for i, (w, _) in enumerate(self.word_counter.most_common())
        }
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.num_vocab = len(self.word2idx)

        self._id_tokens = [self.word2idx[w] for w in self.tokens]
        self.i_idx, self.j_idx, self.xij = list(), list(), list()
        self._create_coocurrence_matrix()

        print("# of words: {}".format(len(self.tokens)))
        print("Vocabulary length: {}".format(self.num_vocab))

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self.window_size, 0)
            end_i = min(i + self.window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j - i)

        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self.i_idx.append(w)
                self.j_idx.append(c)
                self.xij.append(v)

        self.i_idx = torch.LongTensor(self.i_idx)
        self.j_idx = torch.LongTensor(self.j_idx)
        self.xij = torch.FloatTensor(self.xij)

    def get_batches(self, batch_size):
        rand_ids = torch.LongTensor(
            np.random.choice(len(self.xij), len(self.xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p + batch_size]
            yield self.xij[batch_ids], self.i_idx[batch_ids], self.j_idx[
                batch_ids]
