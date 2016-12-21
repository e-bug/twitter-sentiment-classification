# -*- coding: utf-8 -*-
"""remove pound sign in front of words."""

import re
import numpy as np


replacement_patterns = [
        (r"#(\w*)", "\g<1>"),
]


class RegexpReplacer(object):
    
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, line):
        s = line

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        
        return s
