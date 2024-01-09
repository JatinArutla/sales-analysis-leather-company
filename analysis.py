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



st.title('Sales Analysis')



df = pd.read_csv('orders.csv')
df['date'] = pd.to_datetime(df['date'])
df.drop_duplicates(inplace=True)
df.rename(columns={'quantity': 'Units', 'reference': 'SKU Reference', 'title': 'Product Name', 'price_inc': 'Revenue (£)', 'attribute_summary': 'Size'}, inplace=True)

orig_df = pd.read_csv('orders.csv')
orig_df['date'] = pd.to_datetime(orig_df['date'])
df.drop_duplicates(inplace=True)
orig_df.rename(columns={'quantity': 'Units', 'reference': 'SKU Reference', 'title': 'Product Name', 'price_inc': 'Revenue (£)', 'attribute_summary': 'Size'}, inplace=True)

today = datetime.datetime.now()
prev_year = today.year - 3
next_year = today.year - 1
jan_1 = datetime.date(prev_year, 1, 1)
dec_31 = datetime.date(next_year, 12, 31)

d = st.date_input(
    "Select dates",
    (),
    jan_1,
    dec_31,
    format="DD.MM.YYYY",
    key=101,
)

d2 = ()

if(len(d) > 1):
    df = df[(df['date'] >= pd.to_datetime(d[0])) & (df['date'] <= pd.to_datetime(d[1]))].drop(['Unnamed: 0'], axis=1)

    filters_check = st.checkbox('Enable filters')

    if filters_check == True:

        df['Product Name'] = df['Product Name'].apply(str)
        df['Product Name'] = df['Product Name'].str.strip()
        titles_arr = df['Product Name'].sort_values()
        titles_arr = titles_arr.unique().tolist()
        titles_arr = [''] + titles_arr

        df['model'] = df['model'].apply(str)
        df['model'] = df['model'].str.strip()
        models_arr = df['model'].sort_values()
        models_arr = models_arr.unique().tolist()
        models_arr = [''] + models_arr

        df['SKU Reference'] = df['SKU Reference'].apply(str)
        df['SKU Reference'] = df['SKU Reference'].str.strip()
        references_arr = df['SKU Reference'].sort_values()
        references_arr = references_arr.unique().tolist()
        references_arr = [''] + references_arr

        total_options_arr = titles_arr + models_arr + references_arr

        options = st.selectbox('Search (Product Name, Model or SKU Reference)', options=total_options_arr)

        if (options != ''):
            df = df[(df == options).any(axis=1)]    

        brand_sel_col, category_sel_col, sub_category_sel_col, colour_sel_col, size_sel_col = st.columns(5)

        brand_arr = df['manufacturer_name'].unique().tolist()
        brand_arr = np.sort(brand_arr).tolist()
        brand_arr = ['All brands'] + brand_arr
        brand_options = brand_sel_col.selectbox('Select a brand', options=brand_arr)
        if (brand_options == 'All brands'):
            df = df
        else:
            df = df[df['manufacturer_name'] == brand_options]


        category_arr = df['Category'].unique().tolist()
        category_arr = np.sort(category_arr).tolist()
        category_arr = ['All categories'] + category_arr
        category_options = category_sel_col.selectbox('Select a category', options=category_arr)
        if (category_options == 'All categories'):
            df = df
        else:
            df = df[df['Category'] == category_options]


        sub_category_arr = df['T_Cat'].unique().tolist()
        if(np.NaN in sub_category_arr):
            sub_category_arr.remove(np.NaN)
        sub_category_arr = np.sort(sub_category_arr).tolist()
        sub_category_arr = ['All sub categories'] + sub_category_arr
        sub_category_options = sub_category_sel_col.selectbox('Select a sub category', options=sub_category_arr)
        if (sub_category_options == 'All sub categories'):
            df = df
        else:
            df = df[df['customs_description'] == sub_category_options]


        colour_arr = df['colour'].unique().tolist()
        colour_arr = np.sort(colour_arr).tolist()
        colour_arr = ['All colours'] + colour_arr
        colour_options = colour_sel_col.selectbox('Select a colour', options=colour_arr)
        if (colour_options == 'All colours'):
            df = df
        else:
            df = df[df['colour'] == colour_options]

        
        size_arr = df['Size'].unique().tolist()
        size_arr = np.sort(size_arr).tolist()
        size_arr = ['All sizes'] + size_arr
        size_options = size_sel_col.selectbox('Select a size', options=size_arr)
        if (size_options == 'All sizes'):
            df = df
        else:
            df = df[df['Size'] == size_options]

        d2 = st.date_input(
            "Comparison dates",
            (),
            jan_1,
            dec_31,
            format="DD.MM.YYYY",
            key=102,
        )

        if (df['Revenue (£)'].min() < df['Revenue (£)'].max()):
            price_range = st.slider(
            'Select a range of product prices',
            df['Revenue (£)'].min(), df['Revenue (£)'].max(), (df['Revenue (£)'].min(), df['Revenue (£)'].max()))
            if(price_range[0] < price_range[1]):
                df = df[(df['Revenue (£)'] >= price_range[0]) & (df['Revenue (£)'] <= price_range[1])]
            else:
                df = df[df['Revenue (£)'] == price_range[0]]

    df['Revenue (£)'] = df['Revenue (£)'].astype(float)
    refunded_df = df[df['order_state'] == 'Order Refunded']
    dispatched_df = df[(df['order_state'] == 'Order Dispatched') | (df['order_state'] == 'Order Refunded')]
    refunded_df.rename(columns={'Units': 'Units Refunded', 'Revenue (£)': 'Total Refund (£)'}, inplace=True)

    if (len(d2) == 2):
        orig_df['Product Name'] = orig_df['Product Name'].apply(str)
        orig_df['titProduct Namele'] = orig_df['Product Name'].str.strip()
        orig_df['model'] = orig_df['model'].apply(str)
        orig_df['model'] = orig_df['model'].str.strip()
        orig_df['SKU Reference'] = orig_df['SKU Reference'].apply(str)
        orig_df['SKU Reference'] = orig_df['SKU Reference'].str.strip()

        df2 = orig_df[(orig_df['date'] >= pd.to_datetime(d2[0])) & (orig_df['date'] <= pd.to_datetime(d2[1]))].drop(['Unnamed: 0'], axis=1)

        if (options != ''):
            df2 = df2[(df2 == options).any(axis=1)]

        if (brand_options == 'All brands'):
            df2 = df2
        else:
            df2 = df2[df2['manufacturer_name'] == brand_options]

        if (category_options == 'All categories'):
            df2 = df2
        else:
            df2 = df2[df2['Category'] == category_options]

        if (sub_category_options == 'All sub categories'):
            df2 = df2
        else:
            df2 = df2[df2['customs_description'] == sub_category_options]

        if (colour_options == 'All colours'):
            df2 = df2
        else:
            df2 = df2[df2['colour'] == colour_options]

        if (size_options == 'All sizes'):
            df2 = df2
        else:
            df2 = df2[df2['Size'] == size_options]

        if (df2['Revenue (£)'].min() < df2['Revenue (£)'].max()):
            if(price_range[0] < price_range[1]):
                df2 = df2[(df2['Revenue (£)'] >= price_range[0]) & (df2['Revenue (£)'] <= price_range[1])]
            else:
                df2 = df2[df2['Revenue (£)'] == price_range[0]]
        
        df2['Revenue (£)'] = df2['Revenue (£)'].astype(float)
        refunded_df2 = df2[df2['order_state'] == 'Order Refunded']
        refunded_df2.rename(columns={'Units': 'Units Refunded', 'Revenue (£)': 'Total Refund (£)'}, inplace=True)
        dispatched_df2 = df2[(df2['order_state'] == 'Order Dispatched') | (df2['order_state'] == 'Order Refunded')]


    if(d2 == ()):
        c1, c2, c3, c4 = st.columns(4)

        c1 = c1.container(border=True)
        c1.markdown(f'<p class="small-font">Total Revenue</p>', unsafe_allow_html=True)
        c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["Revenue (£)"].sum(), 2)):,}</strong></p>', unsafe_allow_html=True)

        c2 = c2.container(border=True)
        c2.markdown(f'<p class="small-font">Units Sold</p>', unsafe_allow_html=True)
        c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong></p>', unsafe_allow_html=True)

        c3 = c3.container(border=True)
        c3.markdown(f'<p class="small-font">Total Refund</p>', unsafe_allow_html=True)
        c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["Total Refund (£)"].sum(), 2)):,}</strong></p>', unsafe_allow_html=True)

        c4 = c4.container(border=True)
        c4.markdown(f'<p class="small-font">Units Refunded</p>', unsafe_allow_html=True)
        c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong></p>', unsafe_allow_html=True)


    if (len(d2) == 2):
        c1, c2, c3, c4 = st.columns(4)

        p1 = percentage_change(np.round(dispatched_df['Revenue (£)'].sum(), 2), np.round(dispatched_df2['Revenue (£)'].sum(), 2))
        c1 = c1.container(border=True)
        c1.markdown(f'<p class="small-font">Total Revenue</p>', unsafe_allow_html=True)
        if p1 > 0:
            c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["Revenue (£)"].sum(), 2)):,}</strong><span style="color: green;"> (+{p1}%)</span></p>', unsafe_allow_html=True)
        elif p1 == 0:
            c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["Revenue (£)"].sum(), 2)):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["Revenue (£)"].sum(), 2)):,}</strong><span style="color: red;"> ({p1}%)</span></p>', unsafe_allow_html=True)

        p2 = percentage_change(dispatched_df['Units'].sum(), dispatched_df2['Units'].sum())
        c2 = c2.container(border=True)
        c2.markdown(f'<p class="small-font">Units Sold</p>', unsafe_allow_html=True)
        if p2 > 0:
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong><span style="color: green;"> (+{p2}%)</span></p>', unsafe_allow_html=True)
        elif p2 == 0:
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong><span style="color: red;"> ({p2}%)</span></p>', unsafe_allow_html=True)

        p3 = percentage_change((np.round(refunded_df['Total Refund (£)'].sum(), 2)), (np.round(refunded_df2['Total Refund (£)'].sum(), 2)))
        c3 = c3.container(border=True)
        c3.markdown(f'<p class="small-font">Total Refund</p>', unsafe_allow_html=True)
        if p3 > 0:
            c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["Total Refund (£)"].sum(), 2)):,}</strong><span style="color: green;"> (+{p3}%)</span></p>', unsafe_allow_html=True)
        elif p3 == 0:
            c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["Total Refund (£)"].sum(), 2)):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["Total Refund (£)"].sum(), 2)):,}</strong><span style="color: red;"> ({p3}%)</span></p>', unsafe_allow_html=True)

        p4 = percentage_change(refunded_df['Units Refunded'].sum(), refunded_df2['Units Refunded'].sum())
        c4 = c4.container(border=True)
        c4.markdown(f'<p class="small-font">Units Refunded</p>', unsafe_allow_html=True)
        if p4 > 0:
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong><span style="color: green;"> (+{p4}%)</span></p>', unsafe_allow_html=True)
        elif p4 == 0:
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong><span style="color: red;"> ({p4}%)</span></p>', unsafe_allow_html=True)


    dispatched_one_cat_df = dispatched_df.groupby('F_Cat')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
    dispatched_rev_one_cat_df = dispatched_df.groupby('F_Cat')['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
    dispatched_rev_one_cat_df['Revenue (£)'] = np.round(dispatched_rev_one_cat_df['Revenue (£)'], 0)
    dispatched_one_cat_df = pd.merge(dispatched_one_cat_df, dispatched_rev_one_cat_df, how="outer", on="F_Cat")
    refunded_one_cat_df = refunded_df.groupby('F_Cat')['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
    refunded_rev_one_cat_df = refunded_df.groupby('F_Cat')['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)
    refunded_rev_one_cat_df['Total Refund (£)'] = np.round(refunded_rev_one_cat_df['Total Refund (£)'], 0)
    refunded_one_cat_df = pd.merge(refunded_one_cat_df, refunded_rev_one_cat_df, how="outer", on="F_Cat")
    dispatched_one_cat_df = pd.merge(dispatched_one_cat_df, refunded_one_cat_df, how="outer", on="F_Cat")
    dispatched_one_cat_df.replace(np.NaN, 0, inplace=True)
    selection = dataframe_with_selections(dispatched_one_cat_df, 1)

    if (selection['selected_rows_indices'] != []):
        selected_prod = dispatched_one_cat_df.loc[selection['selected_rows_indices'][0]]['F_Cat']
        dispatched_df = dispatched_df[dispatched_df['F_Cat'] == selected_prod]
        refunded_df = refunded_df[refunded_df['F_Cat'] == selected_prod]

        dispatched_two_cat_df = dispatched_df.groupby('S_Cat')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
        dispatched_rev_two_cat_df = dispatched_df.groupby('S_Cat')['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
        refunded_two_cat_df = refunded_df.groupby('S_Cat')['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
        refunded_rev_two_cat_df = refunded_df.groupby('S_Cat')['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)
        
        if(len(dispatched_two_cat_df) != 0):
            dispatched_rev_two_cat_df['Revenue (£)'] = np.round(dispatched_rev_two_cat_df['Revenue (£)'], 0)
            dispatched_two_cat_df = pd.merge(dispatched_two_cat_df, dispatched_rev_two_cat_df, how="outer", on="S_Cat")
            refunded_rev_two_cat_df['Total Refund (£)'] = np.round(refunded_rev_two_cat_df['Total Refund (£)'], 0)
            refunded_two_cat_df = pd.merge(refunded_two_cat_df, refunded_rev_two_cat_df, how="outer", on="S_Cat")
            dispatched_two_cat_df = pd.merge(dispatched_two_cat_df, refunded_two_cat_df, how="outer", on="S_Cat")
            dispatched_two_cat_df.replace(np.NaN, 0, inplace=True)

            selection2 = dataframe_with_selections(dispatched_two_cat_df, 2)

            if(selection2['selected_rows_indices'] != []):
                selected_prod = dispatched_two_cat_df.loc[selection2['selected_rows_indices'][0]]['S_Cat']
                dispatched_df = dispatched_df[dispatched_df['S_Cat'] == selected_prod]
                refunded_df = refunded_df[refunded_df['S_Cat'] == selected_prod]
                three_cat_df = dispatched_df.groupby('T_Cat')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)

                if(len(three_cat_df) == 0):
                    product_df = dispatched_df.groupby('Product Name')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                    
                    dispatched_product_three_cat_df = dispatched_df.groupby('Product Name')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                    dispatched_product_rev_three_cat_df = dispatched_df.groupby('Product Name')['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
                    refunded_product_three_cat_df = refunded_df.groupby('Product Name')['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
                    refunded_product_rev_three_cat_df = refunded_df.groupby('Product Name')['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)
                    
                    dispatched_product_rev_three_cat_df['Revenue (£)'] = np.round(dispatched_product_rev_three_cat_df['Revenue (£)'], 0)
                    dispatched_product_three_cat_df = pd.merge(dispatched_product_three_cat_df, dispatched_product_rev_three_cat_df, how="outer", on="Product Name")
                    refunded_product_rev_three_cat_df['Total Refund (£)'] = np.round(refunded_product_rev_three_cat_df['Total Refund (£)'], 0)
                    refunded_product_three_cat_df = pd.merge(refunded_product_three_cat_df, refunded_product_rev_three_cat_df, how="outer", on="Product Name")
                    dispatched_product_three_cat_df = pd.merge(dispatched_product_three_cat_df, refunded_product_three_cat_df, how="outer", on="Product Name")
                    dispatched_product_three_cat_df.replace(np.NaN, 0, inplace=True)
                    
                    selection4 = dataframe_with_selections(dispatched_product_three_cat_df, 4)
                    if(selection4['selected_rows_indices'] != []):
                        selected_prod = dispatched_product_three_cat_df.loc[selection4['selected_rows_indices'][0]]['Product Name']
                        
                        dispatched_df = dispatched_df[dispatched_df['Product Name'] == selected_prod]
                        refunded_df = refunded_df[refunded_df['Product Name'] == selected_prod]

                        dispatched_sku_three_cat_df = dispatched_df.groupby(['Size', 'SKU Reference'])['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                        dispatched_sku_rev_three_cat_df = dispatched_df.groupby(['Size', 'SKU Reference'])['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
                        refunded_sku_three_cat_df = refunded_df.groupby(['Size', 'SKU Reference'])['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
                        refunded_sku_rev_three_cat_df = refunded_df.groupby(['Size', 'SKU Reference'])['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)

                        dispatched_sku_rev_three_cat_df['Revenue (£)'] = np.round(dispatched_sku_rev_three_cat_df['Revenue (£)'], 0)
                        dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, dispatched_sku_rev_three_cat_df, how="outer", on=['Size', 'SKU Reference'])
                        refunded_sku_rev_three_cat_df['Total Refund (£)'] = np.round(refunded_sku_rev_three_cat_df['Total Refund (£)'], 0)
                        refunded_sku_three_cat_df = pd.merge(refunded_sku_three_cat_df, refunded_sku_rev_three_cat_df, how="outer", on=['Size', 'SKU Reference'])
                        dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, refunded_sku_three_cat_df, how="outer", on=['Size', 'SKU Reference'])
                        dispatched_sku_three_cat_df.replace(np.NaN, 0, inplace=True)
                        temp2_df = dispatched_sku_three_cat_df['Size'].str.split(': ', expand=True)
                        temp2_df.columns = ['F_Size', 'Size']
                        dispatched_sku_three_cat_df['SKU Reference'] = dispatched_sku_three_cat_df['SKU Reference'].astype(str)
                        dispatched_sku_three_cat_df['Size'] = temp2_df['Size']

                        st.dataframe(dispatched_sku_three_cat_df, use_container_width=True)

                else:
                    dispatched_three_cat_df = dispatched_df.groupby('T_Cat')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                    dispatched_rev_three_cat_df = dispatched_df.groupby('T_Cat')['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
                    refunded_three_cat_df = refunded_df.groupby('T_Cat')['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
                    refunded_rev_three_cat_df = refunded_df.groupby('T_Cat')['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)
                    
                    dispatched_rev_three_cat_df['Revenue (£)'] = np.round(dispatched_rev_three_cat_df['Revenue (£)'], 0)
                    dispatched_three_cat_df = pd.merge(dispatched_three_cat_df, dispatched_rev_three_cat_df, how="outer", on="T_Cat")
                    refunded_rev_three_cat_df['Total Refund (£)'] = np.round(refunded_rev_three_cat_df['Total Refund (£)'], 0)
                    refunded_three_cat_df = pd.merge(refunded_three_cat_df, refunded_rev_three_cat_df, how="outer", on="T_Cat")
                    dispatched_three_cat_df = pd.merge(dispatched_three_cat_df, refunded_three_cat_df, how="outer", on="T_Cat")
                    dispatched_three_cat_df.replace(np.NaN, 0, inplace=True)
                    selection3 = dataframe_with_selections(dispatched_three_cat_df, 3)

                    if(selection3['selected_rows_indices'] != []):
                        selected_prod = dispatched_three_cat_df.loc[selection3['selected_rows_indices'][0]]['T_Cat']
                        dispatched_df = dispatched_df[dispatched_df['T_Cat'] == selected_prod]
                        refunded_df = refunded_df[refunded_df['T_Cat'] == selected_prod]

                        dispatched_product_three_cat_df = dispatched_df.groupby('Product Name')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                        dispatched_product_rev_three_cat_df = dispatched_df.groupby('Product Name')['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
                        refunded_product_three_cat_df = refunded_df.groupby('Product Name')['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
                        refunded_product_rev_three_cat_df = refunded_df.groupby('Product Name')['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)
                        
                        dispatched_product_rev_three_cat_df['Revenue (£)'] = np.round(dispatched_product_rev_three_cat_df['Revenue (£)'], 0)
                        dispatched_product_three_cat_df = pd.merge(dispatched_product_three_cat_df, dispatched_product_rev_three_cat_df, how="outer", on="Product Name")
                        refunded_product_rev_three_cat_df['Total Refund (£)'] = np.round(refunded_product_rev_three_cat_df['Total Refund (£)'], 0)
                        refunded_product_three_cat_df = pd.merge(refunded_product_three_cat_df, refunded_product_rev_three_cat_df, how="outer", on="Product Name")
                        dispatched_product_three_cat_df = pd.merge(dispatched_product_three_cat_df, refunded_product_three_cat_df, how="outer", on="Product Name")
                        dispatched_product_three_cat_df.replace(np.NaN, 0, inplace=True)

                        selection5 = dataframe_with_selections(dispatched_product_three_cat_df, 5)
                        if(selection5['selected_rows_indices'] != []):
                            selected_prod = dispatched_product_three_cat_df.loc[selection5['selected_rows_indices'][0]]['Product Name']
                            
                            dispatched_df = dispatched_df[dispatched_df['Product Name'] == selected_prod]
                            refunded_df = refunded_df[refunded_df['Product Name'] == selected_prod]

                            dispatched_sku_three_cat_df = dispatched_df.groupby(['Size', 'SKU Reference'])['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                            dispatched_sku_rev_three_cat_df = dispatched_df.groupby(['Size', 'SKU Reference'])['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
                            refunded_sku_three_cat_df = refunded_df.groupby(['Size', 'SKU Reference'])['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
                            refunded_sku_rev_three_cat_df = refunded_df.groupby(['Size', 'SKU Reference'])['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)

                            dispatched_sku_rev_three_cat_df['Revenue (£)'] = np.round(dispatched_sku_rev_three_cat_df['Revenue (£)'], 0)
                            dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, dispatched_sku_rev_three_cat_df, how="outer", on=['Size', 'SKU Reference'])
                            refunded_sku_rev_three_cat_df['Total Refund (£)'] = np.round(refunded_sku_rev_three_cat_df['Total Refund (£)'], 0)
                            refunded_sku_three_cat_df = pd.merge(refunded_sku_three_cat_df, refunded_sku_rev_three_cat_df, how="outer", on=['Size', 'SKU Reference'])
                            dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, refunded_sku_three_cat_df, how="outer", on=['Size', 'SKU Reference'])
                            dispatched_sku_three_cat_df.replace(np.NaN, 0, inplace=True)
                            temp2_df = dispatched_sku_three_cat_df['Size'].str.split(': ', expand=True)
                            temp2_df.columns = ['F_Size', 'Size']
                            dispatched_sku_three_cat_df['SKU Reference'] = dispatched_sku_three_cat_df['SKU Reference'].astype(str)
                            dispatched_sku_three_cat_df['Size'] = temp2_df['Size']

                            st.dataframe(dispatched_sku_three_cat_df, use_container_width=True)

        else:
            
            dispatched_product_two_cat_df = dispatched_df.groupby('Product Name')['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
            dispatched_product_rev_two_cat_df = dispatched_df.groupby('Product Name')['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
            refunded_product_two_cat_df = refunded_df.groupby('Product Name')['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
            refunded_product_rev_two_cat_df = refunded_df.groupby('Product Name')['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)
            
            dispatched_product_rev_two_cat_df['Revenue (£)'] = np.round(dispatched_product_rev_two_cat_df['Revenue (£)'], 0)
            dispatched_product_two_cat_df = pd.merge(dispatched_product_two_cat_df, dispatched_product_rev_two_cat_df, how="outer", on="Product Name")
            refunded_product_rev_two_cat_df['Total Refund (£)'] = np.round(refunded_product_rev_two_cat_df['Total Refund (£)'], 0)
            refunded_product_two_cat_df = pd.merge(refunded_product_two_cat_df, refunded_product_rev_two_cat_df, how="outer", on="Product Name")
            dispatched_product_two_cat_df = pd.merge(dispatched_product_two_cat_df, refunded_product_two_cat_df, how="outer", on="Product Name")
            dispatched_product_two_cat_df.replace(np.NaN, 0, inplace=True)

            selection4 = dataframe_with_selections(dispatched_product_two_cat_df, 4)
            if(selection4['selected_rows_indices'] != []):
                selected_prod = dispatched_product_two_cat_df.loc[selection4['selected_rows_indices'][0]]['Product Name']

                dispatched_df = dispatched_df[dispatched_df['Product Name'] == selected_prod]
                refunded_df = refunded_df[refunded_df['Product Name'] == selected_prod]

                dispatched_sku_two_cat_df = dispatched_df.groupby(['Size', 'SKU Reference'])['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
                dispatched_sku_rev_two_cat_df = dispatched_df.groupby(['Size', 'SKU Reference'])['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
                refunded_sku_two_cat_df = refunded_df.groupby(['Size', 'SKU Reference'])['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
                refunded_sku_rev_two_cat_df = refunded_df.groupby(['Size', 'SKU Reference'])['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)

                dispatched_sku_rev_two_cat_df['Revenue (£)'] = np.round(dispatched_sku_rev_two_cat_df['Revenue (£)'], 0)
                dispatched_sku_two_cat_df = pd.merge(dispatched_sku_two_cat_df, dispatched_sku_rev_two_cat_df, how="outer", on=['Size', 'SKU Reference'])
                refunded_sku_rev_two_cat_df['Total Refund (£)'] = np.round(refunded_sku_rev_two_cat_df['Total Refund (£)'], 0)
                refunded_sku_two_cat_df = pd.merge(refunded_sku_two_cat_df, refunded_sku_rev_two_cat_df, how="outer", on=['Size', 'SKU Reference'])
                dispatched_sku_two_cat_df = pd.merge(dispatched_sku_two_cat_df, refunded_sku_two_cat_df, how="outer", on=['Size', 'SKU Reference'])
                dispatched_sku_two_cat_df.replace(np.NaN, 0, inplace=True)
                temp2_df = dispatched_sku_two_cat_df['Size'].str.split(': ', expand=True)
                temp2_df.columns = ['F_Size', 'Size']
                dispatched_sku_two_cat_df['SKU Reference'] = dispatched_sku_two_cat_df['SKU Reference'].astype(str)
                dispatched_sku_two_cat_df['Size'] = temp2_df['Size']

                st.dataframe(dispatched_sku_two_cat_df, use_container_width=True)