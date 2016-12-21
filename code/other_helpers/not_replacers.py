# -*- coding: utf-8 -*-
"""insert NOT_ in front of word if preceded by 'not' and remove 'not'."""

import re
import numpy as np


replacement_patterns = [
        (r" not (\w+)", " NOT_\g<1>"),
]


class RegexpReplacer(object):
    
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, line):
        s = line

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        
        return s
