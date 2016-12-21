# -*- coding: utf-8 -*-
"""group emoticons."""

import re
import numpy as np


replacement_patterns = [
        (r":\s?-?\s?\)", ":)"),
        (r":\s?-?\s?d", ":D"),
        (r":\s?-?\s?\(", ":("),
        (r":\s?-?\s?o", ":o"),
        (r"8\s?-?\s?o", "8O"),
        (r":\s?-?\s?\?", ":?"),
        (r"8\s?-?\s?\)", "8)"),
        (r":\s?-?\s?x", ":x"),
        (r":\s?-?\s?p", ":P"),
        (r":\s?-?\s?\|", ":|"),
        (r";\s?-?\s?\)", ";)"),
        (r";\s?-?\s?d", ";D"),
        (r":\s?-?\s?\s", ":S"),
        (r":\s?-?\s?3", ":3"),
        (r":\s?-?\s?\*", ":*"),
        (r"x\s?-?\s?\)", "x)"),

        (r"\(\s?-?\s?:", ":)"),
        (r"\)\s?-?\s?:", ":("),
        (r"o\s?-?\s?:", ":o"),
        (r"o\s?-?\s?8", "8O"),
        (r"\(\s?-?\s?8", "8)"), 
        (r"s\s?-?\s?\:", ":S"),
        (r"\*\s?-?\s?:", ":*"),
        (r"\(\s?-?\s?x", "x)"),

        (r">\s?.?\s?<", ">.<"),
        (r"o\s?.?\s?o", "o.o"),
]


class RegexpReplacer(object):
    
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, line):
        s = line

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        
        return s
