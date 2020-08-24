import streamlit as st
import numpy as np
import pandas as pd

st.title('Welcome to sushi recommender! :sushi:')

CSV_IN = "data/sushi_corr-min2.csv"
MIN_COMMON_ITEMS = 2


df = pd.read_csv(CSV_IN, delimiter=',', skiprows=0, header=0)
df.index = df.columns


def predict_scores(df_sim, ser_target):
    ret = {}
    for item1 in df_sim.index:  # not yet rated by the target user
        v1 = df_sim.loc[item1]

        if v1.notnull().sum() < MIN_COMMON_ITEMS:
            continue
        v11 = v1[v1.notnull()]
        t11 = ser_target[v1.notnull()]
        pred1 = (v11 * t11).sum() / np.abs(v11).sum()

        ret[item1] = pred1

    ser_ret = pd.Series(ret)

    return ser_ret.sort_values(ascending=False)


def get_recomm_by_item_sim(df, target_dic):
    ser_target = pd.Series(target_dic)
    df_scores = df[ser_target.index]
    df_scores = df_scores.drop(index=ser_target.index)
    recomm = predict_scores(df_scores, ser_target)

    return recomm


target_dic = {}
USER_DEFAULT_COUNT = 3

USER_INPUT_COUNT = st.slider(
    'How many sushi you would like to rate?', min_value=MIN_COMMON_ITEMS, max_value=10, value=USER_DEFAULT_COUNT, step=1)


for i in range(USER_INPUT_COUNT):
    st.subheader(f':chopsticks: Sushi {i+1}')
    option = st.selectbox('Name:', (df.columns), key=i)
    rate = st.number_input(
        'Your sushi ratings (0-5):', value=0, min_value=0, max_value=5, step=1, key=i)
    target_dic[option] = rate

ok_button = st.button('OK')

if ok_button:
    recomm = get_recomm_by_item_sim(df, target_dic)
    st.write('\n-----------------------------------------\n')
    st.success('Your recommendation is found! :star::star::star:')
    st.write('Number of items calculated:', len(recomm))
    st.write('Top 5 Recommendation:')
    st.dataframe(recomm.head())
    st.write('\n-----------------------------------------\n')
