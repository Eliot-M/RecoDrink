# --- Import packages --- #

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from flask import Flask
from flask import render_template, session, request


df_beers = pd.read_csv('data/detail_df.csv')
df_rating = pd.read_csv('data/rating_df.csv')


# - Change the table format : from tidy to double entry - #
df_pivot = df_rating.pivot_table(index='user_id', columns='unique_id', values='review_overall').fillna(0)

# - Get overall top rated beers (for default selection)
df_beers_top = df_beers.sort_values("review_overall_mean", ascending=False)

# - Get list of all beers
beer_ls = list(set(df_beers['beer_brew_name']))
beer_ls.sort()

#
