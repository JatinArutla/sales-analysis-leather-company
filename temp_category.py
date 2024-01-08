import streamlit as st
import altair as alt
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
import seaborn as sns
import datetime
import re

pd.options.mode.chained_assignment = None

def dataframe_with_selections(df, inp_key):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=False)},
        disabled=df.columns,
        key=inp_key,
        use_container_width=True,
    )
    selected_indices = list(np.where(edited_df.Select)[0])
    selected_rows = df[edited_df.Select]
    return {"selected_rows_indices": selected_indices, "selected_rows": selected_rows}

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def percentage_change(orig, new):
    p = ((orig - new) / new) * 100
    return np.round(p, 1)

def percentage_change_df(df, orig, new, col_name):
    df[orig] = df[orig].replace(np.nan, 0)
    df[new] = df[new].replace(np.nan, 0)
    df[col_name] = ((df[orig] - df[new]) / df[new]) * 100
    # df[col_name] = ((df[orig] - df[new]) / ((df[new] + df[orig]) / 2)) * 100
    df[col_name] = df[col_name].round(1)
    df[col_name] = df[col_name].replace(np.inf, '')
    df[col_name] = df[col_name].replace(np.nan, 0)
    df[orig] = df[orig].astype(str) + ' (' + df[col_name].astype(str) + '%)'
    df[orig] = df[orig].str.replace('(%)','')
    df[orig] = df[orig].str.replace('.0','')
    return df


st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
.small-font {
    font-size:16px !important;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv('orders.csv')
df['date'] = pd.to_datetime(df['date'])

orig_df = pd.read_csv('orders.csv')
orig_df['date'] = pd.to_datetime(orig_df['date'])

# today = datetime.datetime.now()
# prev_year = today.year - 3
# next_year = today.year - 1
# jan_1 = datetime.date(prev_year, 1, 1)
# dec_31 = datetime.date(next_year, 12, 31)

st.title('Sales Analysis')

# date_options_arr = ['None', 'Today', 'Yesterday', 'Last Week', 'This Month', 'Last Month', 'This Year', 'Last Year']
# date_options = st.selectbox('Date range options', options=date_options_arr)

# d = st.date_input(
#     "Select dates",
#     (),
#     jan_1,
#     dec_31,
#     format="DD.MM.YYYY",
#     key=1,
# )

one_cat_df = df.groupby('F_Cat')['quantity'].count().reset_index()
selection = dataframe_with_selections(one_cat_df, 1)

if (selection['selected_rows_indices'] != []):
    selected_prod = one_cat_df.loc[selection['selected_rows_indices'][0]]['F_Cat']
    df = df[df['F_Cat'] == selected_prod]
    two_cat_df = df.groupby('S_Cat')['quantity'].count().reset_index()
    selection2 = dataframe_with_selections(two_cat_df, 2)

    if(selection2['selected_rows_indices'] != []):
        selected_prod = two_cat_df.loc[selection2['selected_rows_indices'][0]]['S_Cat']
        df = df[df['S_Cat'] == selected_prod]
        three_cat_df = df.groupby('T_Cat')['quantity'].count().reset_index()
        selection3 = dataframe_with_selections(three_cat_df, 3)