# --- Import packages --- #
from scripts.__init__ import *  # import pandas, numpy, scipy, flask

# --- Functions --- #

# - ###################
# - Get information - #


def row_from_id(df, beer_id, col_id='unique_id'):
    """ Function to get information about beers from an id.

    :param df: DataFrame with beers characteristics
    :param beer_id: id of the beer-brewery or list of ids
    :param col_id: name of the columns which contain the unique beer-brewery id
    :return: all columns for the associated row, in dataframe format
    """

    if isinstance(id, int):
        df_row = df.loc[df[col_id] == beer_id].drop_duplicates(keep='first')  # if only 1 id
        return df_row

    elif isinstance(id, list):
        df_row = df.loc[df[col_id].isin(beer_id)].drop_duplicates(keep='first')
        return df_row
    else:
        print('Error to deal with, one day')


def id_from_fullname(df, name, col_name='beer_brew_name', col_id='unique_id'):
    """ Function to get the id of a beer from its name.

    :param df: DataFrame with beers characteristics
    :param name: name of the beer-brewery from user selection
    :param col_name: name of the columns which contain the beer name
    :param col_id: name of the columns which contain the beer id
    :return: id (int) or associated list of ids
    """

    if isinstance(name, str):
        id_beer = df.loc[df[col_name] == name, col_id].iloc[0]
        return id_beer

    elif isinstance(name, list):
        id_beer = list(set(df.loc[df[col_name].isin(name), col_id]))
        return id_beer
    else:
        print('Error to deal with, one day')
        return -1


# - #########################################
# - Create new information and formatting - #


def create_user_id(df_matrix):
    """ Function to create a new user id for the current user.

    Note: user_id was created based on a range(). Then a simple increment is sufficient.

    :param df_matrix: user_beer matrix in pandas df format. Have users ids as index
    :return: a new id based on an increment
    """

    new_user = max(df_matrix.index) + 1
    return new_user


def user_selection2row(selection_beer, selection_rating, df_matrix, df_beers):
    """ Function to transform list of beers and rating to compliant rows for matrices.


    :param selection_beer: list of beers selected by the current user
    :param selection_rating: list of ratings for selected beers by the current user
    :param df_matrix: user_beer matrix in pandas df format. Have users ids as index
    :param df_beers: beer DataFrame in pandas df format. Have users ids as index
    :return: tuple of 3 elements: list, pandas series (both for matrices format) and user_id
    """

    # Convert beers names from interface selection to ids
    # /!\ on peut retirer cette étape si on stock des tuple (nom,id) derrière la liste de l'interface
    idx = id_from_fullname(df=df_beers, name=selection_beer)

    # "One_hot" encoding based on columns
    selection_index = [df_matrix.columns.get_loc(x) for x in idx]
    user_list = [0] * len(df_matrix.columns)

    for index, rating_value in zip(selection_index, selection_rating):
        user_list[index] = rating_value

    # Create a pandas series based on the previous list
    user_series = pd.Series(user_list, index=df_matrix.columns)
    user_series.name = create_user_id(df_matrix=df_matrix)

    # Return tuple with both format and the user id
    return user_list, user_series, user_series.name


# - #######################
# - Data Transformation - #


def close_users(user_to_compare, df_matrix, top_to_select=500):
    """ Get closest users in terms of common beers tested.

    :param user_to_compare: list of beers tested by current user in matrix format
    :param df_matrix: user_beer matrix in pandas df format. Have users ids as index
    :param top_to_select: number of closest users to keep
    :return: users ids for closes users selected
    """

    non_zero_idx_user = [idx for idx, val in enumerate(user_to_compare) if val != 0]  # indexes with a rating

    # Get users with the most common beers (sum all ratings (binary) for indexes tested by the current user
    series_intersect = df_matrix[df_matrix.columns[non_zero_idx_user]].astype(bool).sum(axis=1)  # common beers count
    series_intersect = series_intersect.sort_values(ascending=False)  # sort for selection

    return list(series_intersect.iloc[0:top_to_select].index)


def filter_matrix(df_matrix, users_to_keep, series_user_to_add):
    """ Remove unwanted users and add the current user in the matrix for analysis.

    :param df_matrix: user_beer matrix in pandas df format. Have users ids as index
    :param users_to_keep: list of ids for users to keep from the function "close_users"
    :param series_user_to_add: pandas series of beers tested by current user. User id as series name
    :return: updated matrix with new user and its closest users from initial data
    """

    # Filter matrix with pre-selected users. Slightly better perf than a .isin()
    df_matrix = df_matrix.loc[np.in1d(df_matrix.index, users_to_keep)]

    # Add current user.
    df_matrix = df_matrix.append(series_user_to_add)

    # Filter columns. Unnecessary since a sparse matrix is used after.
    # matrix_pd = matrix_pd.loc[:, matrix_pd.sum(axis=0) > 0]

    return df_matrix


def top_df(df, num_recommendations=20):
    """ Return the firsts rows of a DataFrame

    :param df: Any pandas DataFrame
    :param num_recommendations: number of recommendations to return for the current user. Default is 20
    :return: Top X row of the DataFrame
    """
    return df.head(num_recommendations)


# - ########################
# - Prediction functions - #


def preds_from_ratings(df_matrix):
    """ Create prediction values for each beer and each user.

    :param df_matrix: user_beer matrix in pandas df format. Have users ids as index
    :return: user_beer matrix with all predict ratings in DataFrame format. Have users ids as index
    """

    # Change format to improve perf of the SVD function.
    mat_user_beer = df_matrix.as_matrix()
    mat_user_beer = csr_matrix(mat_user_beer, dtype=float)  # sparse format

    # Singular Value Decomposition
    mat_u, sigma, mat_vt = svds(mat_user_beer, k=10)

    # Prediction for users
    df_predictions = pd.DataFrame(np.dot(np.dot(mat_u, np.diag(sigma)), mat_vt), columns=df_matrix.columns)

    return df_predictions


def recommend_beers(df_predictions, df_beers, user_series, num_recommendations=20):
    """ Recommend beers according to highest prediction values from "preds_from_ratings" for the current user.

    Note: If there is no beer rated by the current user it return 20 beers with the highest rating.

    :param df_predictions: user_beer matrix with predict ratings in DataFrame format. Have users ids as index
    :param df_beers: DataFrame with beers characteristics
    :param user_series: pandas series of beers tested by current user. User id as series name
    :param num_recommendations: number of recommendations to return for the current user. Default is 20
    :return: DataFrame with beers characteristics for the X most recommended beers
    """

    # Get and sort the predictions for the current user
    user_pred_sort = df_predictions.iloc[-1].sort_values(ascending=False)  # the current user is the last one

    # Remove beers already rated, and get a top X
    user_rated_beers = list(user_series[user_series > 0].index)
    id_beers_to_reco = list(user_pred_sort[~user_pred_sort.index.isin(user_rated_beers)].
                            iloc[:num_recommendations].index)

    # Get information for these beers
    df_beers = df_beers.loc[np.in1d(df_beers.unique_id, id_beers_to_reco)]

    # Force the order in the DataFrame (based on predictions order)
    df_beers['unique_id'] = pd.Categorical(df_beers['unique_id'], id_beers_to_reco)

    # Return beers information with correct order
    return df_beers.sort_values("unique_id")


# - ########################
# - Agregated functions - #


def updated_list(input_list, selection_beer):
    """ Update the selection list by removing already selected beers

    :param input_list:
    :param selection_beer:
    :return:
    """

    # Set difference between these lists
    output_list = list(set(input_list) - set(selection_beer))

    # Sort back the list
    output_list.sort()

    return output_list


def launch_reco(selection_beer, df_beers=df_beers, df_matrix=df_pivot):
    """ All steps needed to perform a recommendation from some beers

    :param selection_beer:
    :param df_beers:
    :param df_matrix:
    :return:
    """
    beers_ls = [x[0] for x in selection_beer]
    ratings_ls = [x[1] for x in selection_beer]

    user_l, user_s, user_ident = user_selection2row(selection_beer=beers_ls, selection_rating=ratings_ls,
                                                    df_matrix=df_matrix, df_beers=df_beers)

    # Get close users based on similar consumed beers, by default top 500 users (should be enough).
    users = close_users(user_to_compare=user_l, df_matrix=df_matrix)

    # Filter the main matrix to improve computation performances
    new_df = filter_matrix(df_matrix=df_matrix, users_to_keep=users, series_user_to_add=user_s)

    # Predict a rating for all beers of all users
    df_preds = preds_from_ratings(df_matrix=new_df)

    # Return top beers for the current user. It not include the ones already rated by the user.
    out = recommend_beers(df_predictions=df_preds, df_beers=df_beers, user_series=user_s, num_recommendations=20)

    return out, user_ident, beers_ls

# - end functions
