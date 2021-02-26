# -*- coding: utf-8 -*-
"""
Code to calculate basic NLP measures from a speech excerpt, as in Morgan et al 2021:
https://doi.org/10.1101/2021.01.04.20248717
Please cite the paper above if you use this code for your own work.
Author: Dr Sarah E Morgan, 21/02/2021
"""

import nltk
import string


# count the number of sentences in a text:
def get_no_sent(text):
    text = remove_text_inside_brackets(text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return len(sentences)

# count the number of words in a text:
def get_no_words(text):
    text = remove_text_inside_brackets(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t not in string.punctuation] # removes punctuation, so punctuation is not counted as a word
    return len(tokens)

# returns the total no. of words, total no. of sentences and mean sentence length for a text:
def meas_basic(text):
    totalnowords = get_no_words(text)
    totalnosent = get_no_sent(text)
    meansent = totalnowords/totalnosent

    return totalnowords, totalnosent, meansent

# removes text inside brackets, e.g. [?], [inaudible]
# taken from: https://stackoverflow.com/questions/14596884/remove-text-between-and
def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)