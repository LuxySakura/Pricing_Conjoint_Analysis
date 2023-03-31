import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def utility(x, *params):
    """
    Calculate the utility function using combination and parameters

    Notes:
    ------
    - The basic combination is: ($6.99, HD, some content restricted, support devices: 1, support download: 1, Ads)
    - The sequence of combination is

    :param x: Specific combination
    :param b_ad: Equals to 1 when "No Ads"
    :param b_dow_6: Equals to 1 when "Max download 6"
    :param b_dow_2: Equals to 1 when "Max download 2"
    :param b_dev_4: Equals to 1 when "Max devices 4"
    :param b_dev_2: Equals to 1 when "Max devices 2"
    :param b_high: Equals to 1 when "$19.99"
    :param b_med: Equals to 1 when "$15.49"
    :param b_low: Equals to 1 when "$9.99"
    :param b_unlimited: b_ultra: Equals to 1 when "unlimited content"
    :param b_ultra: Equals to 1 when the video quality is "Ultra HD"
    :param b_0: Utility for basic combination
    :param b_full: Equals to 1 when the video quality is "Full HD"
    :return: The utility of certain combinations
    :rtype: float
    """
    # util = b_0 + b_full * x[0] + b_ultra * x[1] + \
    #     b_unlimited * x[2] + b_low * x[3] + \
    #     b_med * x[4] + b_high * x[5] + \
    #     b_dev_2 * x[6] + b_dev_4 * x[7] + \
    #     b_dow_2 * x[8] + b_dow_6 * x[9] + b_ad * x[10]
    b, b_0 = params
    util = np.dot(x, b) + b_0
    return util


if __name__ == "__main__":
    # Read the .csv file
    df = pd.read_csv("Survey Result.csv")

    # Turn combination to numeric
    # c = [Full, Ultra, Unlimited, low, med, high, 2, 4, 2, 6, No Ads]
    c_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    arr_1 = np.tile(c_1, (36, 1))
    c_2 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    arr_2 = np.tile(c_2, (36, 1))

    c_3 = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
    arr_3 = np.tile(c_3, (36, 1))
    c_4 = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    arr_4 = np.tile(c_4, (36, 1))
    c_5 = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    arr_5 = np.tile(c_5, (36, 1))

    c_6 = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1])
    arr_6 = np.tile(c_6, (36, 1))
    c_7 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    arr_7 = np.tile(c_7, (36, 1))
    c_8 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1])
    arr_8 = np.tile(c_8, (36, 1))

    c_9 = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    arr_9 = np.tile(c_9, (36, 1))
    c_10 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    arr_10 = np.tile(c_10, (36, 1))
    c_11 = np.array([1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    arr_11 = np.tile(c_11, (36, 1))

    # Read each combination's rating and get the average of them
    comb_1 = df["$6.99 per month: HD, Limited content, max devices: 1; max download: 1; Have Ads "] \
        .astype(float)
    comb_2 = df["$6.99 per month: HD, Limited content, max devices: 2; max download: 1; Have Ads "] \
        .astype(float)

    comb_3 = df["$9.99 per month: HD, Unlimited content, max devices: 1; max download: 2; Have Ads "] \
        .astype(float)
    comb_4 = df["$9.99 per month: Full HD, Limited content, max devices: 2; max download: 1; No Ads "] \
        .astype(float)
    comb_5 = df["$9.99 per month: Ultra HD, Limited content, max devices: 4; max download: 2; Have Ads "] \
        .astype(float)

    comb_6 = df["$15.49 per month: Full HD, Unlimited content, max devices: 1; max download: 6; No Ads "] \
        .astype(float)
    comb_7 = df["$15.49 per month: Full HD, Unlimited content, max devices: 2; max download: 1; Have Ads "] \
        .astype(float)
    comb_8 = df["$15.49 per month: Ultra HD, Limited content, max devices: 4; max download: 2; No Ads "] \
        .astype(float)

    comb_9 = df["$19.99 per month: Ultra HD, Unlimited content, max devices: 2 max download: 6; No Ads "] \
        .astype(float)
    comb_10 = df["$19.99 per month: Ultra HD, Unlimited content, max devices: 4 max download: 6; No Ads "] \
        .astype(float)
    comb_11 = df["$19.99 per month: Full HD, Limited content, max devices: 4 max download: 2; No Ads "] \
        .astype(float)

    # Build x data
    combinations = np.vstack((arr_1, arr_2, arr_3, arr_4,
                              arr_5, arr_6, arr_7, arr_8,
                              arr_9, arr_10, arr_11))

    # Build y data
    ratings = np.concatenate([comb_1, comb_2, comb_3, comb_4,
                              comb_5, comb_6, comb_7, comb_8,
                              comb_9, comb_10, comb_11], axis=0)

    # Linear Regression Model
    model = LinearRegression()

    model.fit(combinations, ratings)
    print("The coefficient we got is:\n")
    print(model.coef_)
    print("The Intercept we got is:\n")
    print(model.intercept_)
