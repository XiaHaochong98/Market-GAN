import numpy as np


class Tokenizer():
    def __init__(self, num_words):
        self.num_words = num_words
        self.word_index = {}
        self.word_counts = {}
        self.index_docs = {}
        self.index_word = {}

    def fit_on_text(self, text):
        for word in text:
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index) + 1
                self.index_word[len(self.word_index)] = word
                self.word_counts[word] = 1
            else:
                self.word_counts[word] += 1

    def word_to_one_hot(self, word):
        one_hot = np.zeros(self.num_words)
        if word in self.word_index:
            one_hot[self.word_index[word] - 1] = 1.0
        return one_hot

    def one_hot_to_word(self, one_hot):
        # for each sample in one_hot, find the index of the max value
        # and return the word corresponding to that index
        res = []
        # print(self.index_word)
        for sample in one_hot:
            # print(sample)
            # print(np.argmax(sample))
            res.append(self.index_word[np.argmax(sample) + 1])
            # print(res)
        return res
