import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time

from utils import (get_context, create_neg_sample_list, get_negatives,
                   PreprocessData, Word2VecIterableDataset, calc_spearman_coef,
                   preprocess_raw_text, save_pickle, load_pickle)
from models import SkipGramModel


def train(dataset, loader, embed_dim, batch_size, max_epoch, window_size,
          neg_sample_size, lr, test_file):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # pickle file name
    LOSS_EMBED = f'loss_{embed_dim}.npy'
    SPEARMAN_EMBED = f'spearman_{embed_dim}.npy'
    LOSS_HISTORY_PICKLE = f'loss_history_{embed_dim}.pkl'
    SPEARMAN_HISTORY_PICKLE = f'spearman_history_{embed_dim}.pkl'

    # initialize variables.
    loss_history = []
    spear_history = []
    best_spearman = 0.0
    worst_loss = 100

    # model and optimizer
    model = SkipGramModel(dataset.num_vocab, embed_dim)
    model.to(DEVICE)
    optimizer = optim.SparseAdam(model.parameters(), lr=lr)

    # create negatives table
    negatives = create_neg_sample_list(dataset.word_count_filtered)

    for epoch in range(max_epoch):
        for idx, batch in enumerate(loader):
            batch_loss = []
            start_time = time.time()

            # concat N sentences into matrix for vectorized training.
            target, context, neg_sample = [], [], []
            for sentence in batch:
                # split the words in a sentence
                train_words = sentence.rstrip().split(' ')

                # subsampling
                train_words = [
                    word for word in train_words
                    if random.random() < dataset.drop_prob.get(word, 1.0)
                ]

                # remove any sentence that is shorter than 5 words.
                if len(train_words) <= 5:
                    continue

                train_words_idx = [
                    dataset.word2idx.get(word, 0) for word in train_words
                ]

                # get target and context words
                for i, target_word in enumerate(train_words_idx):
                    context_words = get_context(train_words_idx, i,
                                                window_size)
                    context.extend(context_words)
                    target.extend([target_word] * len(context_words))

            # negative sampling
            neg_sample = get_negatives(target, negatives, len(target), K=5)
            # unigram sampling
            # neg_sample = np.random.randint(low=0,
            #                                high=dataset.num_vocab,
            #                                size=(len(target), neg_sample_size),
            #                                dtype=int)

            # convert to tensor
            target_idx = torch.LongTensor(target).to(DEVICE)
            context_idx = torch.LongTensor(context).to(DEVICE)
            neg_idx = torch.LongTensor(np.vstack(neg_sample)).to(DEVICE)

            optimizer.zero_grad()
            loss = model.forward(target_idx, context_idx, neg_idx).to(DEVICE)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # spearman after each batch
            if test_file:
                new_spearman = calc_spearman_coef(
                    test_file, dataset.word2idx,
                    model.wi.weight.cpu().data.numpy())
                spear_history.append(new_spearman)

                if new_spearman > best_spearman:
                    model.save_embedding(dataset.idx2word, SPEARMAN_EMBED)
                    best_spearman = new_spearman
            else:
                new_spearman = 0.0

            time_taken = time.time() - start_time
            print(
                "Epoch: {}/{} \t Batch: {} \t Loss: {}  \t Spearman: {}  \t Time_taken: {} seconds"
                .format(epoch + 1, max_epoch, idx * batch_size,
                        round(loss.item(), 5), round(new_spearman, 5),
                        round(time_taken, 5)))

    save_pickle(LOSS_HISTORY_PICKLE, loss_history)
    save_pickle(SPEARMAN_HISTORY_PICKLE, spear_history)

    print('Best loss: ', min(loss_history))
    print('Best spearman score: ', max(spear_history))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="word2vec")
    PARSER.add_argument("--data",
                        required=True,
                        help="Train data with in text file.")
    PARSER.add_argument("--preprocess",
                        help="Pickle file for PreprocessData class object.")
    PARSER.add_argument("--embed-dim",
                        type=int,
                        default=100,
                        help="Embedding dimension.")
    PARSER.add_argument("--batch-size",
                        type=int,
                        default=10,
                        help="Batch size.")
    PARSER.add_argument("--epoch",
                        type=int,
                        default=1,
                        help="Number of epoch for training.")
    PARSER.add_argument("--lr",
                        type=float,
                        default=0.01,
                        help="Learning rate.")
    PARSER.add_argument("--window-size",
                        type=int,
                        default=2,
                        help="Window size for context word")
    PARSER.add_argument("--min-freq",
                        type=int,
                        default=5,
                        help="Minimum word frequency to include in vocab.")
    PARSER.add_argument("--neg-sample-size",
                        type=int,
                        default=5,
                        help="Number of negative sample.")
    PARSER.add_argument("--wordsim", help="CSV for wordsim evaluation data.")
    ARGS = PARSER.parse_args()

    if ARGS.preprocess:
        dataset = load_pickle(ARGS.preprocess)
    else:
        print("creating dataset object....")
        dataset = PreprocessData(ARGS.data, min_count=ARGS.min_freq)
        save_pickle('dataset.pkl', dataset)

    # create dataloader
    iterable_dataset = Word2VecIterableDataset(ARGS.data)
    loader = DataLoader(iterable_dataset,
                        batch_size=ARGS.batch_size,
                        drop_last=False)

    train(dataset=dataset,
          loader=loader,
          embed_dim=ARGS.embed_dim,
          batch_size=ARGS.batch_size,
          max_epoch=ARGS.epoch,
          window_size=ARGS.window_size,
          neg_sample_size=ARGS.neg_sample_size,
          lr=ARGS.lr,
          test_file=ARGS.wordsim)