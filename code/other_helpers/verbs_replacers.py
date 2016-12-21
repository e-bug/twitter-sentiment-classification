# -*- coding: utf-8 -*-
"""fix most common typos and splits most common abbreaviations."""


import re
import numpy as np


replacement_patterns = [
        (r"won\'t", "will not"),
        (r"can\'t", "cannot"),
        (r"ain\'t", "is not"),
        (r"(\w+)\'ll", "\g<1> will"),
        (r"(\w+)n\'t", "\g<1> not"),
        (r"i\'m", "i am"),
        (r"(\w+)\'ve", "\g<1> have"),
        (r"(\w+)\'re", "\g<1> are"),
        (r"(\w+)\'d", "\g<1> would"),
# people might not put ' and verbs finishing in nt should be separated
        (r"wont", "will not"),
        (r"cant", "cannot"),
        (r"aint", "is not"),
        (r"dont", "do not"),
        (r"didnt", "did not"),
        (r"mightnt", "might not"),
        (r"maynt", "may not"),
        (r"couldnt", "could not"),
        (r"wouldnt", "would not"),
        (r"shouldnt", "should not"),
        (r"wasnt", "was not"),
        (r"werent", "were not"),
# typos with "be" and "have"
        (r"im", "i am"),
        (r"ive", "i have"),
        (r"youre", "you are"),
        (r"youve", "you have"),
        (r"it\'s", "it is"),
        (r"he\'s", "he is"),
        (r"hes", "he is"),
        (r"she\'s", "she is"),
        (r"shes", "she is"),
        (r"were", "we are"),
        (r"weve", "we have"),
        (r"theyre", "they are"),
        (r"theyve", "they have"),
]


class RegexpReplacer(object):
    
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, line):
        s = line

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        
        return s
