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

today = datetime.datetime.now()
prev_year = today.year - 3
next_year = today.year - 1
jan_1 = datetime.date(prev_year, 1, 1)
dec_31 = datetime.date(next_year, 12, 31)

st.title('Sales Analysis')

# date_options_arr = ['None', 'Today', 'Yesterday', 'Last Week', 'This Month', 'Last Month', 'This Year', 'Last Year']
# date_options = st.selectbox('Date range options', options=date_options_arr)

d = st.date_input(
    "Select dates",
    (),
    jan_1,
    dec_31,
    format="DD.MM.YYYY",
    key=1,
)

d2 = ()

if(d != () and len(d) > 1):
    df = df[(df['date'] >= pd.to_datetime(d[0])) & (df['date'] <= pd.to_datetime(d[1]))].drop(['Unnamed: 0'], axis=1)

    filters_check = st.checkbox('Enable filters')

    if filters_check == True:

        df['title'] = df['title'].apply(str)
        df['title'] = df['title'].str.strip()
        titles_arr = df['title'].sort_values()
        titles_arr = titles_arr.unique().tolist()
        titles_arr = [''] + titles_arr

        df['model'] = df['model'].apply(str)
        df['model'] = df['model'].str.strip()
        models_arr = df['model'].sort_values()
        models_arr = models_arr.unique().tolist()
        models_arr = [''] + models_arr

        df['reference'] = df['reference'].apply(str)
        df['reference'] = df['reference'].str.strip()
        references_arr = df['reference'].sort_values()
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

        
        size_arr = df['attribute_summary'].unique().tolist()
        size_arr = np.sort(size_arr).tolist()
        size_arr = ['All sizes'] + size_arr
        size_options = size_sel_col.selectbox('Select a size', options=size_arr)
        if (size_options == 'All sizes'):
            df = df
        else:
            df = df[df['attribute_summary'] == size_options]

        d2 = st.date_input(
            "Comparison dates",
            (),
            jan_1,
            dec_31,
            format="DD.MM.YYYY",
            key=2,
        )

        if (df['price_inc'].min() < df['price_inc'].max()):
            price_range = st.slider(
            'Select a range of product prices',
            df['price_inc'].min(), df['price_inc'].max(), (df['price_inc'].min(), df['price_inc'].max()))
            if(price_range[0] < price_range[1]):
                df = df[(df['price_inc'] >= price_range[0]) & (df['price_inc'] <= price_range[1])]
            else:
                df = df[df['price_inc'] == price_range[0]]


    df['price_inc'] = df['price_inc'].astype(float)
    refunded_df = df[df['order_state'] == 'Order Refunded']
    dispatched_df = df[(df['order_state'] == 'Order Dispatched') | (df['order_state'] == 'Order Refunded')]
    dispatched_df['price_inc'] = dispatched_df['price_inc'].astype(float)

    if (len(d2) == 2):
        orig_df['title'] = orig_df['title'].apply(str)
        orig_df['title'] = orig_df['title'].str.strip()
        orig_df['model'] = orig_df['model'].apply(str)
        orig_df['model'] = orig_df['model'].str.strip()
        orig_df['reference'] = orig_df['reference'].apply(str)
        orig_df['reference'] = orig_df['reference'].str.strip()

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
            df2 = df2[df2['attribute_summary'] == size_options]

        if (df2['price_inc'].min() < df2['price_inc'].max()):
            if(price_range[0] < price_range[1]):
                df2 = df2[(df2['price_inc'] >= price_range[0]) & (df2['price_inc'] <= price_range[1])]
            else:
                df2 = df2[df2['price_inc'] == price_range[0]]
        
        df2['price_inc'] = df2['price_inc'].astype(float)
        refunded_df2 = df2[df2['order_state'] == 'Order Refunded']
        dispatched_df2 = df2[(df2['order_state'] == 'Order Dispatched') | (df2['order_state'] == 'Order Refunded')]
        dispatched_df2['price_inc'] = dispatched_df2['price_inc'].astype(float)

    
    if(d2 == ()):
        c1, c2, c3, c4 = st.columns(4)

        c1 = c1.container(border=True)
        c1.markdown(f'<p class="small-font">Total Revenue</p>', unsafe_allow_html=True)
        c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["price_inc"].sum(), 2)):,}</strong></p>', unsafe_allow_html=True)

        c2 = c2.container(border=True)
        c2.markdown(f'<p class="small-font">Units Sold</p>', unsafe_allow_html=True)
        c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["fifo_quantity"].sum()):,}</strong></p>', unsafe_allow_html=True)

        c3 = c3.container(border=True)
        c3.markdown(f'<p class="small-font">Total Refund</p>', unsafe_allow_html=True)
        c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["price_inc"].sum(), 2)):,}</strong></p>', unsafe_allow_html=True)

        c4 = c4.container(border=True)
        c4.markdown(f'<p class="small-font">Units Refunded</p>', unsafe_allow_html=True)
        c4.markdown(f'<p class="big-font"><strong>{(refunded_df["fifo_quantity"].sum()):,}</strong></p>', unsafe_allow_html=True)


    if (len(d2) == 2):
        c1, c2, c3, c4 = st.columns(4)

        p1 = percentage_change(np.round(dispatched_df['price_inc'].sum(), 2), np.round(dispatched_df2['price_inc'].sum(), 2))
        c1 = c1.container(border=True)
        c1.markdown(f'<p class="small-font">Total Revenue</p>', unsafe_allow_html=True)
        if p1 > 0:
            c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["price_inc"].sum(), 2)):,}</strong><span style="color: green;"> (+{p1}%)</span></p>', unsafe_allow_html=True)
        elif p1 == 0:
            c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["price_inc"].sum(), 2)):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c1.markdown(f'<p class="big-font">£<strong>{(np.round(dispatched_df["price_inc"].sum(), 2)):,}</strong><span style="color: red;"> ({p1}%)</span></p>', unsafe_allow_html=True)

        p2 = percentage_change(dispatched_df['fifo_quantity'].sum(), dispatched_df2['fifo_quantity'].sum())
        c2 = c2.container(border=True)
        c2.markdown(f'<p class="small-font">Units Sold</p>', unsafe_allow_html=True)
        if p2 > 0:
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["fifo_quantity"].sum()):,}</strong><span style="color: green;"> (+{p2}%)</span></p>', unsafe_allow_html=True)
        elif p2 == 0:
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["fifo_quantity"].sum()):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["fifo_quantity"].sum()):,}</strong><span style="color: red;"> ({p2}%)</span></p>', unsafe_allow_html=True)

        p3 = percentage_change((np.round(refunded_df['price_inc'].sum(), 2)), (np.round(refunded_df2['price_inc'].sum(), 2)))
        c3 = c3.container(border=True)
        c3.markdown(f'<p class="small-font">Total Refund</p>', unsafe_allow_html=True)
        if p3 > 0:
            c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["price_inc"].sum(), 2)):,}</strong><span style="color: green;"> (+{p3}%)</span></p>', unsafe_allow_html=True)
        elif p3 == 0:
            c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["price_inc"].sum(), 2)):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c3.markdown(f'<p class="big-font">£<strong>{(np.round(refunded_df["price_inc"].sum(), 2)):,}</strong><span style="color: red;"> ({p3}%)</span></p>', unsafe_allow_html=True)

        p4 = percentage_change(refunded_df['fifo_quantity'].sum(), refunded_df2['fifo_quantity'].sum())
        c4 = c4.container(border=True)
        c4.markdown(f'<p class="small-font">Units Refunded</p>', unsafe_allow_html=True)
        if p4 > 0:
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["fifo_quantity"].sum()):,}</strong><span style="color: green;"> (+{p4}%)</span></p>', unsafe_allow_html=True)
        elif p4 == 0:
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["fifo_quantity"].sum()):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
        else:
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["fifo_quantity"].sum()):,}</strong><span style="color: red;"> ({p4}%)</span></p>', unsafe_allow_html=True)


    # show_df = dispatched_df.groupby('title')['fifo_quantity'].agg(Units='count').sort_values(by=['Units'], ascending=False).reset_index()
    # show_rev_df = dispatched_df.groupby('title')['price_inc'].agg(Revenue='sum').sort_values(by=['Revenue'], ascending=False).reset_index()
    show_df = dispatched_df.groupby(['title', 'model'])['fifo_quantity'].count().reset_index().rename(columns={'fifo_quantity': 'Units'}).sort_values(by=['Units'], ascending=False)
    show_rev_df = dispatched_df.groupby(['title', 'model'])['price_inc'].sum().reset_index().rename(columns={'price_inc': 'Revenue'}).sort_values(by=['Revenue'], ascending=False)
    show_rev_df['Revenue'] = np.round(show_rev_df['Revenue'], 0)
    show_rev_df = show_rev_df.rename(columns={'Revenue': 'Revenue (£)'})
    show_df = pd.merge(show_df, show_rev_df, how="outer", on="title")

    # show_refunded_df = refunded_df.groupby('title')['fifo_quantity'].agg(Refunded_Units='count').sort_values(by=['Refunded_Units'], ascending=False).reset_index()
    # show_rev_refunded_df = refunded_df.groupby('title')['price_inc'].agg(Refunded_Revenue='sum').sort_values(by=['Refunded_Revenue'], ascending=False).reset_index()
    show_refunded_df = refunded_df.groupby(['title', 'model'])['fifo_quantity'].count().reset_index().rename(columns={'fifo_quantity': 'Refunded_Units'}).sort_values(by=['Refunded_Units'], ascending=False)
    show_rev_refunded_df = refunded_df.groupby(['title', 'model'])['price_inc'].sum().reset_index().rename(columns={'price_inc': 'Refunded_Revenue'}).sort_values(by=['Refunded_Revenue'], ascending=False)
    show_rev_refunded_df['Refunded_Revenue'] = np.round(show_rev_refunded_df['Refunded_Revenue'], 0)
    show_refunded_df = pd.merge(show_refunded_df, show_rev_refunded_df, how="outer", on="title")
    show_refunded_df = show_refunded_df.rename(columns={'Refunded_Units': 'Units Refunded', 'Refunded_Revenue': 'Total Refund (£)'})

    show_df = pd.merge(show_df, show_refunded_df, how="outer", on="title")
    # temp_model_df = df[['title', 'model']].drop_duplicates(subset=['model'])
    # show_df = pd.merge(show_df, temp_model_df, how="outer", on="title")
    show_df.replace(np.NaN, 0, inplace=True)
    show_df.rename(columns={'title': 'Product Name', 'model_x_x': 'Model'}, inplace=True)
    show_df = show_df[['Model', 'Product Name', 'Units', 'Revenue (£)', 'Units Refunded', 'Total Refund (£)']]

    if (len(d2) == 2):

        show_df2 = dispatched_df2.groupby(['title', 'model'])['fifo_quantity'].count().reset_index().rename(columns={'fifo_quantity': 'Units'}).sort_values(by=['Units'], ascending=False)
        show_rev_df2 = dispatched_df2.groupby(['title', 'model'])['price_inc'].sum().reset_index().rename(columns={'price_inc': 'Revenue'}).sort_values(by=['Revenue'], ascending=False)
        show_rev_df2['Revenue'] = np.round(show_rev_df2['Revenue'], 0)
        show_rev_df2 = show_rev_df2.rename(columns={'Revenue': 'Revenue (£)'})
        show_df2 = pd.merge(show_df2, show_rev_df2, how="outer", on="title")

        show_refunded_df2 = refunded_df2.groupby(['title', 'model'])['fifo_quantity'].count().reset_index().rename(columns={'fifo_quantity': 'Refunded_Units'}).sort_values(by=['Refunded_Units'], ascending=False)
        show_rev_refunded_df2 = refunded_df2.groupby(['title', 'model'])['price_inc'].sum().reset_index().rename(columns={'price_inc': 'Refunded_Revenue'}).sort_values(by=['Refunded_Revenue'], ascending=False)
        show_rev_refunded_df2['Refunded_Revenue'] = np.round(show_rev_refunded_df2['Refunded_Revenue'], 0)
        show_refunded_df2 = pd.merge(show_refunded_df2, show_rev_refunded_df2, how="outer", on="title")
        show_refunded_df2 = show_refunded_df2.rename(columns={'Refunded_Units': 'Units Refunded', 'Refunded_Revenue': 'Total Refund (£)'})

        show_df2 = pd.merge(show_df2, show_refunded_df2, how="outer", on="title")
        show_df2.replace(np.NaN, 0, inplace=True)
        show_df2.rename(columns={'title': 'Product Name'}, inplace=True)
        show_df2.rename(columns={'title': 'Product Name', 'model_x_x': 'Model'}, inplace=True)

        show_df = pd.merge(show_df, show_df2, how='outer', on=['Product Name', 'Model'])
        show_df = percentage_change_df(show_df, 'Units_x', 'Units_y', 'perc_units')
        show_df = percentage_change_df(show_df, 'Revenue (£)_x', 'Revenue (£)_y', 'perc_revenue')
        show_df = percentage_change_df(show_df, 'Units Refunded_x', 'Units Refunded_y', 'perc_refunded_units')
        show_df = percentage_change_df(show_df, 'Total Refund (£)_x', 'Total Refund (£)_y', 'perc_refunded_revenue')
        show_df.rename(columns={'Units_x': 'Units', 'Revenue (£)_x': 'Revenue (£)', 'Units Refunded_x': 'Units Refunded', 'Total Refund (£)_x': 'Total Refund (£)', 'SKU Reference_x': 'SKU Reference'}, inplace=True)
        show_df = show_df[['Model', 'Product Name', 'Units', 'Revenue (£)', 'Units Refunded', 'Total Refund (£)']]


    selection = dataframe_with_selections(show_df, 'main')

    csv = convert_df(show_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='details_df.csv',
        mime='text/csv',
    )

    table_column, graph_column = st.columns([0.45, 0.55])
    
    if (selection['selected_rows_indices'] != []):
        selected_prod = show_df.loc[selection['selected_rows_indices'][0]]['Product Name']
        selected_df = df[df['title'] == selected_prod]
        selected_df['reference'] = selected_df['reference'].astype(str)
        selected_df['reference'].apply(lambda x: x.replace(',', ''))

        selected_sku_df = selected_df[(selected_df['order_state'] == 'Order Dispatched') | (selected_df['order_state'] == 'Order Refunded')]
        refunded_sku_df = selected_df[selected_df['order_state'] == 'Order Refunded']
        selected_sku_df['price_inc'] = selected_sku_df['price_inc'].astype(float)

        show_sku_df = selected_sku_df.groupby(['attribute_summary', 'reference'])['fifo_quantity'].agg(Units='sum').sort_values(by=['Units'], ascending=False).reset_index()
        show_rev_sku_df = selected_sku_df.groupby(['attribute_summary', 'reference'])['price_inc'].agg(Revenue='sum').sort_values(by=['Revenue'], ascending=False).reset_index()
        show_rev_sku_df['Revenue'] = np.round(show_rev_sku_df['Revenue'], 0)
        show_rev_sku_df = show_rev_sku_df.rename(columns={'Revenue': 'Revenue (£)'})
        show_rev_sku_df.drop(['reference'], axis=1, inplace=True)
        show_sku_df = pd.merge(show_sku_df, show_rev_sku_df, how="outer", on="attribute_summary")


        show_refunded_sku_df = refunded_sku_df.groupby(['attribute_summary', 'reference'])['fifo_quantity'].agg(Refunded_Units='sum').sort_values(by=['Refunded_Units'], ascending=False).reset_index()
        show_rev_refunded_sku_df = refunded_sku_df.groupby(['attribute_summary', 'reference'])['price_inc'].agg(Refunded_Revenue='sum').sort_values(by=['Refunded_Revenue'], ascending=False).reset_index()
        show_rev_refunded_sku_df['Refunded_Revenue'] = np.round(show_rev_refunded_sku_df['Refunded_Revenue'], 0)
        show_rev_refunded_sku_df.drop(['reference'], axis=1, inplace=True)
        show_refunded_sku_df = pd.merge(show_refunded_sku_df, show_rev_refunded_sku_df, on="attribute_summary")
        show_refunded_sku_df = show_refunded_sku_df.rename(columns={'Refunded_Units': 'Units Refunded', 'Refunded_Revenue': 'Total Refund (£)'})
        show_refunded_sku_df.drop(['reference'], axis=1, inplace=True)

        show_sku_df = pd.merge(show_sku_df, show_refunded_sku_df, how="outer", on="attribute_summary")
        show_sku_df.replace(np.NaN, 0, inplace=True)
        show_sku_df.rename(columns={'attribute_summary': 'Size', 'reference': 'SKU Reference'}, inplace=True)
        show_sku_df['SKU Reference'] = show_sku_df['SKU Reference'].astype(str)

        temp_df = show_sku_df['Size'].str.split(': ', expand=True)
        temp_df.columns = ['F_Size', 'Size']
        show_sku_df['Size'] = temp_df['Size']
        
        total_show_sku_df = show_sku_df.copy()
        total_show_sku_df.loc['Total'] = total_show_sku_df.select_dtypes(np.number).sum()

        if (d2 == ()):
            table_column.markdown(f"<p class='big-font'><strong>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}</strong></p>", unsafe_allow_html=True)

            table_column.write(total_show_sku_df)

            csv2 = convert_df(total_show_sku_df)
            table_column.download_button(
                label="Download data as CSV",
                data=csv2,
                file_name=f'sku_{selected_prod}.csv',
                mime='text/csv',
            )

            if (len(show_sku_df) > 2):
                t1, t2, t3 = show_sku_df['Units'].idxmax(), show_sku_df['Units Refunded'].idxmax(), show_sku_df['Units'].idxmin()
                table_column.markdown(f'<p class="big-font"><strong>{selected_prod}</strong></p>', unsafe_allow_html=True)
                if (show_sku_df.iloc[t1]['Units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {show_sku_df.iloc[t1]["Units"].astype(int)} units of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {show_sku_df.iloc[t1]["Units"].astype(int)} unit of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                if (show_sku_df.iloc[t2]['Units Refunded'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {show_sku_df.iloc[t2]["Units Refunded"].astype(int)} units of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Total Refund (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {show_sku_df.iloc[t2]["Units Refunded"].astype(int)} unit of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Total Refund (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                if (show_sku_df.iloc[t3]['Units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {show_sku_df.iloc[t3]["Units"].astype(int)} units of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {show_sku_df.iloc[t3]["Units"].astype(int)} unit of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                # table_column.write(f'{show_sku_df.iloc[t1]['Size']} having an SKU reference of {show_sku_df.iloc[t1]['SKU Reference']} sold the most units: {show_sku_df.iloc[t1]['Units'].astype(int)} with a revenue of £{(show_sku_df.iloc[t1]['Revenue (£)'].astype(int)):,}')
                # table_column.write(f'{show_sku_df.iloc[t2]['Size']} having an SKU reference of {show_sku_df.iloc[t2]['SKU Reference']} had the most units refunded: {show_sku_df.iloc[t2]['Units Refunded'].astype(int)} with refund of £{(show_sku_df.iloc[t2]['Total Refund (£)'].astype(int)):,}')
                # table_column.write(f'{show_sku_df.iloc[t3]['Size']} having an SKU reference of {show_sku_df.iloc[t3]['SKU Reference']} had the least units sold: {show_sku_df.iloc[t3]['Units'].astype(int)} with a revenue of £{(show_sku_df.iloc[t3]['Revenue (£)'].astype(int)):,}')


        if (len(d2) == 2):
            selected_df2 = df2[df2['title'] == selected_prod]
            selected_df2['reference'] = selected_df2['reference'].astype(str)
            selected_df2['reference'].apply(lambda x: x.replace(',', ''))

            selected_sku_df2 = selected_df2[(selected_df2['order_state'] == 'Order Dispatched') | (selected_df2['order_state'] == 'Order Refunded')]
            refunded_sku_df2 = selected_df2[selected_df2['order_state'] == 'Order Refunded']
            selected_sku_df2['price_inc'] = selected_sku_df2['price_inc'].astype(float)

            show_sku_df2 = selected_sku_df2.groupby(['attribute_summary', 'reference'])['fifo_quantity'].agg(Units='sum').sort_values(by=['Units'], ascending=False).reset_index()
            show_rev_sku_df2 = selected_sku_df2.groupby(['attribute_summary', 'reference'])['price_inc'].agg(Revenue='sum').sort_values(by=['Revenue'], ascending=False).reset_index()
            show_rev_sku_df2['Revenue'] = np.round(show_rev_sku_df2['Revenue'], 0)
            show_rev_sku_df2 = show_rev_sku_df2.rename(columns={'Revenue': 'Revenue (£)'})
            show_rev_sku_df2.drop(['reference'], axis=1, inplace=True)
            show_sku_df2 = pd.merge(show_sku_df2, show_rev_sku_df2, how="outer", on="attribute_summary")
            

            show_refunded_sku_df2 = refunded_sku_df2.groupby(['attribute_summary', 'reference'])['fifo_quantity'].agg(Refunded_Units='sum').sort_values(by=['Refunded_Units'], ascending=False).reset_index()
            show_rev_refunded_sku_df2 = refunded_sku_df2.groupby(['attribute_summary', 'reference'])['price_inc'].agg(Refunded_Revenue='sum').sort_values(by=['Refunded_Revenue'], ascending=False).reset_index()
            show_rev_refunded_sku_df2['Refunded_Revenue'] = np.round(show_rev_refunded_sku_df2['Refunded_Revenue'], 0)
            show_rev_refunded_sku_df2.drop(['reference'], axis=1, inplace=True)
            show_refunded_sku_df2 = pd.merge(show_refunded_sku_df2, show_rev_refunded_sku_df2, on="attribute_summary")
            show_refunded_sku_df2 = show_refunded_sku_df2.rename(columns={'Refunded_Units': 'Units Refunded', 'Refunded_Revenue': 'Total Refund (£)'})
            show_refunded_sku_df2.drop(['reference'], axis=1, inplace=True)

            show_sku_df2 = pd.merge(show_sku_df2, show_refunded_sku_df2, how="outer", on="attribute_summary")
            show_sku_df2.replace(np.NaN, 0, inplace=True)
            show_sku_df2.rename(columns={'attribute_summary': 'Size', 'reference': 'SKU Reference'}, inplace=True)
            show_sku_df2['SKU Reference'] = show_sku_df2['SKU Reference'].astype(str)

            temp2_df = show_sku_df2['Size'].str.split(': ', expand=True)
            temp2_df.columns = ['F_Size', 'Size']
            show_sku_df2['Size'] = temp2_df['Size']

            total_show_sku_df2 = show_sku_df2.copy()
            total_show_sku_df2.loc['Total'] = total_show_sku_df2.select_dtypes(np.number).sum()

            temp_show_sku_df = pd.merge(total_show_sku_df, total_show_sku_df2, how='outer', on=['Size'])
            temp_show_sku_df = percentage_change_df(temp_show_sku_df, 'Units_x', 'Units_y', 'perc_units')
            temp_show_sku_df = percentage_change_df(temp_show_sku_df, 'Revenue (£)_x', 'Revenue (£)_y', 'perc_revenue')
            temp_show_sku_df = percentage_change_df(temp_show_sku_df, 'Units Refunded_x', 'Units Refunded_y', 'perc_refunded_units')
            temp_show_sku_df = percentage_change_df(temp_show_sku_df, 'Total Refund (£)_x', 'Total Refund (£)_y', 'perc_refunded_revenue')
            temp_show_sku_df.rename(columns={'Units_x': 'Units', 'Revenue (£)_x': 'Revenue (£)', 'Units Refunded_x': 'Units Refunded', 'Total Refund (£)_x': 'Total Refund (£)', 'SKU Reference_x': 'SKU Reference'}, inplace=True)
            temp_show_sku_df = temp_show_sku_df[['Size', 'SKU Reference', 'Units', 'Revenue (£)', 'Units Refunded', 'Total Refund (£)']]
            
            temp_list = list(temp_show_sku_df.index)
            temp_list[-1] = 'Total'
            temp_show_sku_df.index = temp_list


            table_column.markdown(f"<p class='big-font'><strong>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')} and {d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}</strong></p>", unsafe_allow_html=True)
            table_column.write(temp_show_sku_df)
            # table_column.write(show_sku_df2)

            csv2 = convert_df(temp_show_sku_df)
            table_column.download_button(
                label="Download data as CSV",
                data=csv2,
                file_name=f'sku_{selected_prod}.csv',
                mime='text/csv',
            )

            if (len(show_sku_df) > 2):

                t1, t2, t3 = show_sku_df['Units'].idxmax(), show_sku_df['Units Refunded'].idxmax(), show_sku_df['Units'].idxmin()
                table_column.markdown(f"<p class='big-font'><strong>{selected_prod}</strong></p>", unsafe_allow_html=True)
                table_column.markdown(f"<p class='small-font'><strong>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}</strong></p>", unsafe_allow_html=True)
                if (show_sku_df.iloc[t1]['Units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {show_sku_df.iloc[t1]["Units"].astype(int)} units of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {show_sku_df.iloc[t1]["Units"].astype(int)} unit of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                if (show_sku_df.iloc[t2]['Units Refunded'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {show_sku_df.iloc[t2]["Units Refunded"].astype(int)} units of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Total Refund (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {show_sku_df.iloc[t2]["Units Refunded"].astype(int)} unit of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Total Refund (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                if (show_sku_df.iloc[t3]['Units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {show_sku_df.iloc[t3]["Units"].astype(int)} units of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {show_sku_df.iloc[t3]["Units"].astype(int)} unit of size: {show_sku_df.iloc[t1]["Size"]} with a revenue of £{(show_sku_df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
        
                # table_column.write(f'{show_sku_df.iloc[t1]['Size']} having an SKU reference of {show_sku_df.iloc[t1]['SKU Reference']} sold the most units: {show_sku_df.iloc[t1]['Units'].astype(int)} with a revenue of £{(show_sku_df.iloc[t1]['Revenue (£)'].astype(int)):,}')
                # table_column.write(f'{show_sku_df.iloc[t2]['Size']} having an SKU reference of {show_sku_df.iloc[t2]['SKU Reference']} had the most units refunded: {show_sku_df.iloc[t2]['Units Refunded'].astype(int)} with refund of £{(show_sku_df.iloc[t2]['Total Refund (£)'].astype(int)):,}')
                # table_column.write(f'{show_sku_df.iloc[t3]['Size']} having an SKU reference of {show_sku_df.iloc[t3]['SKU Reference']} had the least units sold: {show_sku_df.iloc[t3]['Units'].astype(int)} with a revenue of £{(show_sku_df.iloc[t3]['Revenue (£)'].astype(int)):,}')


                merged_show_sku_df = pd.merge(show_sku_df, show_sku_df2, how='outer', on=['Size', 'SKU Reference'])
                merged_show_sku_df['merged_units'] = merged_show_sku_df['Units_x'] + merged_show_sku_df['Units_y']
                merged_show_sku_df['merged_revenue'] = merged_show_sku_df['Revenue (£)_x'] + merged_show_sku_df['Revenue (£)_y']
                merged_show_sku_df['merged_refunded_units'] = merged_show_sku_df['Units Refunded_x'] + merged_show_sku_df['Units Refunded_y']
                merged_show_sku_df['merged_refunded_revenue'] = merged_show_sku_df['Total Refund (£)_x'] + merged_show_sku_df['Total Refund (£)_y']

                t1_2, t2_2, t3_2 = merged_show_sku_df['merged_units'].idxmax(), merged_show_sku_df['merged_refunded_units'].idxmax(), merged_show_sku_df['merged_units'].idxmin()
                table_column.markdown(f"<p class='small-font'><strong>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')} and {d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}</strong></p>", unsafe_allow_html=True)
                if(merged_show_sku_df.iloc[t1_2]['merged_units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {merged_show_sku_df.iloc[t1_2]["merged_units"].astype(int)} units of size: {merged_show_sku_df.iloc[t1_2]["Size"]} with a revenue of £{(merged_show_sku_df.iloc[t1_2]["merged_revenue"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {merged_show_sku_df.iloc[t1_2]["merged_units"].astype(int)} unit of size: {merged_show_sku_df.iloc[t1_2]["Size"]} with a revenue of £{(merged_show_sku_df.iloc[t1_2]["merged_revenue"].astype(int)):,}</p>', unsafe_allow_html=True)
                if (merged_show_sku_df.iloc[t2_2]['merged_refunded_units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {merged_show_sku_df.iloc[t2_2]["merged_refunded_units"].astype(int)} units of size: {merged_show_sku_df.iloc[t2_2]["Size"]} with a revenue of £{(merged_show_sku_df.iloc[t2_2]["merged_refunded_revenue"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {merged_show_sku_df.iloc[t2_2]["merged_refunded_units"].astype(int)} unit of size: {merged_show_sku_df.iloc[t2_2]["Size"]} with a revenue of £{(merged_show_sku_df.iloc[t2_2]["merged_refunded_revenue"].astype(int)):,}</p>', unsafe_allow_html=True)
                if (merged_show_sku_df.iloc[t3_2]['merged_units'].astype(int) != 1):
                    table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {merged_show_sku_df.iloc[t3_2]["merged_units"].astype(int)} units of size: {merged_show_sku_df.iloc[t3_2]["Size"]} with a revenue of £{(merged_show_sku_df.iloc[t3_2]["merged_revenue"].astype(int)):,}</p>', unsafe_allow_html=True)
                else:
                    table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {merged_show_sku_df.iloc[t3_2]["merged_units"].astype(int)} unit of size: {merged_show_sku_df.iloc[t3_2]["Size"]} with a revenue of £{(merged_show_sku_df.iloc[t3_2]["merged_revenue"].astype(int)):,}</p>', unsafe_allow_html=True)
                # table_column.write(f'{merged_show_sku_df.iloc[t1_2]['Size']} having an SKU reference of {merged_show_sku_df.iloc[t1_2]['SKU Reference']} sold the most units: {merged_show_sku_df.iloc[t1_2]['merged_units'].astype(int)} with a revenue of £{(merged_show_sku_df.iloc[t1_2]['merged_revenue'].astype(int)):,}</p>', unsafe_allow_html=True)
                # table_column.write(f'{merged_show_sku_df.iloc[t2_2]['Size']} having an SKU reference of {merged_show_sku_df.iloc[t2_2]['SKU Reference']} had the most units refunded: {merged_show_sku_df.iloc[t2_2]['merged_refunded_units'].astype(int)} with refund of £{(merged_show_sku_df.iloc[t2_2]['merged_refunded_revenue'].astype(int)):,}')
                # table_column.write(f'{merged_show_sku_df.iloc[t3_2]['Size']} having an SKU reference of {merged_show_sku_df.iloc[t3_2]['SKU Reference']} had the least units sold: {merged_show_sku_df.iloc[t3_2]['merged_units'].astype(int)} with a revenue of £{(merged_show_sku_df.iloc[t3_2]['merged_revenue'].astype(int)):,}')



        sub_1, sub_2 = graph_column.tabs(["Units", "Revenue"])

        if (len(d2) == 0):
            graph_df = df[df['title'] == selected_prod]
            graph_df = graph_df[(graph_df['order_state'] == 'Order Dispatched') | (graph_df['order_state'] == 'Order Refunded')]
            graph_df.rename(columns={'price_inc': 'Revenue', 'date': 'Date', 'quantity': 'Quantity'}, inplace=True)
            
            rev_df = graph_df[['Date', 'Revenue']].groupby('Date').sum().reset_index()
            # rev_df.rename(columns={'Revenue': f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}"}, inplace=True)
            temp_df = graph_df[['Date', 'Quantity']].groupby('Date').count().reset_index()
            # temp_df.rename(columns={'Quantity': f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}"}, inplace=True)
            temp_df.rename(columns={'Quantity': 'Units'}, inplace=True)

            # sub_1.line_chart(temp_df, x="Date", y=[f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}"], color=["#0000FF"])
            # sub_2.line_chart(rev_df, x="Date", y="Revenue")

            # base = alt.Chart(temp_df).encode(
            #     x='Date',
            #     y='Units',
            # ).interactive()
            # lines = base.mark_line()
            # points = base.mark_point(filled=True)
            # sub_1.altair_chart(lines + points, use_container_width=True)

            chart = alt.Chart(temp_df).mark_point(filled=True).encode(x='Date', y='Units')
            line = alt.Chart(temp_df, title=f'{selected_prod} Units Sold from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}').mark_line().encode(x='Date', y='Units').interactive()
            sub_1.altair_chart(chart + line, use_container_width=True)

            chart = alt.Chart(rev_df).mark_point(filled=True).encode(x='Date', y='Revenue')
            line = alt.Chart(rev_df, title=f'{selected_prod} Revenue from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}').mark_line().encode(x='Date', y='Revenue').interactive()
            sub_2.altair_chart(chart + line, use_container_width=True)

            # base = alt.Chart(rev_df).encode(
            #     x='Date',
            #     y='Revenue',
            # ).interactive()
            # lines = base.mark_line()
            # points = base.mark_point(filled=True)
            # sub_2.altair_chart(lines + points, use_container_width=True)

        
        if (len(d2) == 2):
            graph_df = df[df['title'] == selected_prod]
            graph_df = graph_df[(graph_df['order_state'] == 'Order Dispatched') | (graph_df['order_state'] == 'Order Refunded')]
            graph_df.rename(columns={'price_inc': 'Revenue', 'date': 'Date', 'quantity': 'Quantity'}, inplace=True)
            graph_df2 = df2[df2['title'] == selected_prod]
            graph_df2 = graph_df2[(graph_df2['order_state'] == 'Order Dispatched') | (graph_df2['order_state'] == 'Order Refunded')]
            graph_df2.rename(columns={'price_inc': 'Revenue', 'date': 'Date', 'quantity': 'Quantity'}, inplace=True)

            rev_df = graph_df[['Date', 'Revenue']].groupby('Date').sum().reset_index()
            rev_df2 = graph_df2[['Date', 'Revenue']].groupby('Date').sum().reset_index()
            rev_df['Date_2'] = rev_df2['Date']
            rev_df['Revenue_2'] = rev_df2['Revenue']
            # rev_df.rename(columns={'Revenue': f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}", 'Revenue_2': f"{d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}"}, inplace=True)

            temp_df = graph_df[['Date', 'Quantity']].groupby('Date').count().reset_index()
            temp_df2 = graph_df2[['Date', 'Quantity']].groupby('Date').count().reset_index()
            temp_df['Date_2'] = temp_df['Date']
            temp_df['Quantity_2'] = temp_df2['Quantity']
            # temp_df.rename(columns={'Quantity': f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}", 'Quantity_2': f"{d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}"}, inplace=True)
            

            # base = alt.Chart(temp_df).encode(
            #     x='Date',
            #     # y=['Quantity', 'Quantity_2'],
            # ).repeat(layer=['Quantity', 'Quantity_2']).interactive()
            # lines = base.mark_line()
            # points = base.mark_point(filled=True)
            # sub_1.altair_chart(lines + points, use_container_width=True)

            # base = alt.Chart(rev_df).encode(
            #     x='Date',
            #     # y=['Revenue', 'Revenue_2'],
            # ).repeat(layer=['Revenue', 'Revenue_2']).interactive()
            # lines = base.mark_line()
            # points = base.mark_point(filled=True)
            # sub_2.altair_chart(lines + points, use_container_width=True)



            # base = alt.Chart(temp_df.reset_index()).encode(x='Date')
            # base = alt.layer(
            #     base.mark_line(color='blue').encode(y='Quantity'),
            #     base.mark_line(color='red').encode(y='Quantity_2'),
            #     base.mark_point(color='blue', filled=True).encode(y='Quantity'),
            #     base.mark_point(color='red', filled=True).encode(y='Quantity_2')
            # ).interactive()
            # # lines = base.mark_line()
            # # points = base.mark_point(filled=True)
            # sub_1.altair_chart(base, use_container_width=True)


            # base = alt.Chart(rev_df.reset_index()).encode(x='Date')
            # base = alt.layer(
            #     base.mark_line(color='blue').encode(y='Revenue'),
            #     base.mark_line(color='red').encode(y='Revenue_2'),
            #     base.mark_point(color='blue', filled=True).encode(y='Revenue'),
            #     base.mark_point(color='red', filled=True).encode(y='Revenue_2')
            # ).interactive()
            # # lines = base.mark_line()
            # # points = base.mark_point(filled=True)
            # sub_2.altair_chart(base, use_container_width=True)



            chart1 = alt.Chart(temp_df).mark_point(color='blue', filled=True).encode(x='Date', y='Quantity')
            line1 = alt.Chart(temp_df, title=f'{selected_prod} Units Sold').mark_line(color='blue').encode(x='Date', y='Quantity').interactive()
            chart2 = alt.Chart(temp_df).mark_point(color='red', filled=True).encode(x='Date_2', y='Quantity_2')
            line2 = alt.Chart(temp_df, title=f'{selected_prod} Units Sold').mark_line(color='red').encode(x='Date_2', y='Quantity_2').interactive()
            sub_1.altair_chart(chart1 + line1 + chart2 + line2, use_container_width=True)
            sub_1.markdown(f"<p class='small-font'><span style='color: blue;'>Blue: </span>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}</p>", unsafe_allow_html=True)
            sub_1.markdown(f"<p class='small-font'><span style='color: red;'>Red: </span>{d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}</p>", unsafe_allow_html=True)

            chart1 = alt.Chart(rev_df).mark_point(color='blue', filled=True).encode(x='Date', y='Revenue')
            line1 = alt.Chart(rev_df, title=f'{selected_prod} Revenue').mark_line(color='blue').encode(x='Date', y='Revenue').interactive()
            chart2 = alt.Chart(rev_df).mark_point(color='red', filled=True).encode(x='Date_2', y='Revenue_2')
            line2 = alt.Chart(rev_df, title=f'{selected_prod} Revenue').mark_line(color='red').encode(x='Date_2', y='Revenue_2').interactive()
            sub_2.altair_chart(chart1 + line1 + chart2 + line2, use_container_width=True)
            sub_2.markdown(f"<p class='small-font'><span style='color: blue;'>Blue: </span>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}</p>", unsafe_allow_html=True)
            sub_2.markdown(f"<p class='small-font'><span style='color: red;'>Red: </span>{d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}</p>", unsafe_allow_html=True)



            # fig1 = alt.Chart(rev_df, title=f'{selected_prod} Revenue from {d[0]} to {d[1]}').mark_line().encode(
            #     x='Day')
            # sub_1.altair_chart(fig1, use_container_width=True)
            # sub_1.line_chart(temp_df, x="Date", y=[f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}", f"{d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}"], color=["#FF0000", "#0000FF"])
            
            # fig2 = alt.Chart(temp_df, title=f'{selected_prod} Units Sold from {d[0]} to {d[1]}').mark_line().encode(
            #     x='Day', y='Quantity')
            # sub_2.altair_chart(fig2, use_container_width=True)
            # sub_2.line_chart(rev_df, x="Date", y=[f"{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}", f"{d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}"], color=["#FF0000", "#0000FF"])