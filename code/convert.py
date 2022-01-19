#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv('../data/IMDB.csv')
df.to_csv('../data/convert.csv', sep='\t', index=False)
