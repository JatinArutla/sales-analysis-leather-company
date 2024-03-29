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

def sku_condense_dataframe(dispatched_df, refunded_df):
    dispatched_sku_three_cat_df = dispatched_df.groupby(['Size', 'SKU Reference', 'Channel'])['Units'].sum().reset_index().sort_values(by=['Units'], ascending=False).reset_index(drop=True)
    dispatched_sku_rev_three_cat_df = dispatched_df.groupby(['Size', 'SKU Reference', 'Channel'])['Revenue (£)'].sum().reset_index().sort_values(by=['Revenue (£)'], ascending=False).reset_index(drop=True)
    refunded_sku_three_cat_df = refunded_df.groupby(['Size', 'SKU Reference', 'Channel'])['Units Refunded'].sum().reset_index().sort_values(by=['Units Refunded'], ascending=False).reset_index(drop=True)
    refunded_sku_rev_three_cat_df = refunded_df.groupby(['Size', 'SKU Reference', 'Channel'])['Total Refund (£)'].sum().reset_index().sort_values(by=['Total Refund (£)'], ascending=False).reset_index(drop=True)

    dispatched_sku_rev_three_cat_df['Revenue (£)'] = np.round(dispatched_sku_rev_three_cat_df['Revenue (£)'], 0)
    dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, dispatched_sku_rev_three_cat_df, how="outer", on=['Size', 'SKU Reference', 'Channel'])
    refunded_sku_rev_three_cat_df['Total Refund (£)'] = np.round(refunded_sku_rev_three_cat_df['Total Refund (£)'], 0)
    refunded_sku_three_cat_df = pd.merge(refunded_sku_three_cat_df, refunded_sku_rev_three_cat_df, how="outer", on=['Size', 'SKU Reference', 'Channel'])
    dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, refunded_sku_three_cat_df, how="outer", on=['Size', 'SKU Reference', 'Channel'])
    dispatched_sku_three_cat_df.replace(np.NaN, 0, inplace=True)
    temp2_df = dispatched_sku_three_cat_df['Size'].str.split(': ', expand=True)
    temp2_df.columns = ['F_Size', 'Size']
    dispatched_sku_three_cat_df['SKU Reference'] = dispatched_sku_three_cat_df['SKU Reference'].astype(str)
    dispatched_sku_three_cat_df['Size'] = temp2_df['Size']
    return dispatched_sku_three_cat_df

def product_condense_dataframe(dispatched_df, refunded_df):
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
    return dispatched_product_two_cat_df

def graph_condense(dispatched_df):
    graph_df = dispatched_df[['date', 'Units']].groupby('date').count().reset_index()
    graph_df.rename(columns={'date': 'Date'}, inplace=True)
    graph_df.set_index('Date', inplace=True)
    graph_df = graph_df.asfreq('D')
    graph_df['Units'] = graph_df['Units'].replace(np.nan, 0)
    graph_df.reset_index(inplace=True)
    graph_df['Date'] = graph_df['Date'].astype(str)
    temp = graph_df['Date'].str.split(' ', expand=True)
    temp.columns = ['Date']
    graph_df['Date'] = temp['Date']
    graph_df['Date'] = pd.to_datetime(graph_df['Date'])
    chart = alt.Chart(graph_df).mark_point(filled=True).encode(x='Date', y='Units')
    line = alt.Chart(graph_df, title=f'{selected_prod} Units Sold from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}').mark_line(point=True).encode(x='Date', y='Units').interactive()

    graph_df = graph_df[['Date', 'Units']]
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_point(nearest=True, on='mouseover',
                            fields=['Date'], empty=False)

    # The basic line
    line = alt.Chart(graph_df, title=f'{selected_prod} Units Sold from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}').mark_line().encode(
        x='Date',
        y='Units',
    ).interactive()

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(graph_df).mark_point().encode(
        x='Date',
        y='Units',
        opacity=alt.value(0),
    ).add_params(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(graph_df).mark_rule(color='gray').encode(
        x='Date',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    final = alt.layer(
        line, selectors, points, rules
    ).properties(
        width=600, height=300
    )

    return chart, final

def display_sku(selected_prod, d, d2, dispatched_df, dispatched_sku_three_cat_df):
    table_column, graph_column = st.columns([0.5, 0.5])
    table_column.markdown(f'<p class="big-font"><strong>{selected_prod}</strong></p>', unsafe_allow_html=True)
    table_column.dataframe(dispatched_sku_three_cat_df, use_container_width=True)
    if (len(dispatched_sku_three_cat_df) > 1):
        columns_list = dispatched_sku_three_cat_df.columns
        total_df = pd.DataFrame(columns=columns_list)
        total_df.loc['Total'] = dispatched_sku_three_cat_df.select_dtypes(np.number).sum()
        table_column.dataframe(total_df, use_container_width=True)
        sku_summary(dispatched_sku_three_cat_df, d, d2, table_column)

    chart, line = graph_condense(dispatched_df)
    graph_column.altair_chart(line, use_container_width=True)

def sku_summary(df, d, d2, table_column):
    t1, t2, t3 = df['Units'].idxmax(), df['Units Refunded'].idxmax(), df['Units'].idxmin()
    if(len(d2) <= 1):
        table_column.markdown(f"<p class='small-font'><strong>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')}</strong></p>", unsafe_allow_html=True)
    elif(len(d2) > 1):
        table_column.markdown(f"<p class='small-font'><strong>{d[0].strftime('%d %b %Y')} to {d[1].strftime('%d %b %Y')} and {d2[0].strftime('%d %b %Y')} to {d2[1].strftime('%d %b %Y')}</strong></p>", unsafe_allow_html=True)
    temp_units = df.iloc[t1]['Units'].astype(int)
    if(df.iloc[t1]['Units'].astype(int) > 1):
        table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {df.iloc[t1]["Units"].astype(int)} units of size: {df.iloc[t1]["Size"]} with a revenue of £{(df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
    elif(df.iloc[t1]['Units'].astype(int) == 1):
        table_column.markdown(f'<p class="small-font"><strong>Best Seller:</strong> {df.iloc[t1]["Units"].astype(int)} unit of size: {df.iloc[t1]["Size"]} with a revenue of £{(df.iloc[t1]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
    if (df.iloc[t2]['Units Refunded'].astype(int) > 1):
        table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {df.iloc[t2]["Units Refunded"].astype(int)} units of size: {df.iloc[t2]["Size"]} with a revenue of £{(df.iloc[t2]["Total Refund (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
    elif (df.iloc[t2]['Units Refunded'].astype(int) == 1):
        table_column.markdown(f'<p class="small-font"><strong>Most Refunded:</strong> {df.iloc[t2]["Units Refunded"].astype(int)} unit of size: {df.iloc[t2]["Size"]} with a revenue of £{(df.iloc[t2]["Total Refund (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
    if (temp_units != df.iloc[t3]['Units'].astype(int)):
        if (df.iloc[t3]['Units'].astype(int) > 1):
            table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {df.iloc[t3]["Units"].astype(int)} units of size: {df.iloc[t3]["Size"]} with a revenue of £{(df.iloc[t3]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)
        elif (df.iloc[t3]['Units'].astype(int) == 1):
            table_column.markdown(f'<p class="small-font"><strong>Least Sold:</strong> {df.iloc[t3]["Units"].astype(int)} unit of size: {df.iloc[t3]["Size"]} with a revenue of £{(df.iloc[t3]["Revenue (£)"].astype(int)):,}</p>', unsafe_allow_html=True)

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



# df = pd.read_csv('2024_feb_unduplicated_orders fixed_categories.csv')
df = pd.read_csv('All orders - short custom description.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df.drop_duplicates(inplace=True)
df.rename(columns={'quantity': 'Units', 'reference': 'SKU Reference', 'title': 'Product Name', 'price_inc': 'Revenue (£)', 'attribute_summary': 'Size'}, inplace=True)

stock_df = pd.read_csv('stock_levels_25_mar.csv')
stock_df.rename(columns={'parent_title': 'Product Name', 'stock': 'Stock', 'attribute_summary': 'Size', 'product_url': 'Target page'}, inplace=True)
temp_stock_df = stock_df['Size'].str.split(': ', expand=True)
temp_stock_df.columns = ['F_Size', 'Size']
stock_df['Size'] = temp_stock_df['Size']

# orig_df = pd.read_csv('2024_feb_unduplicated_orders fixed_categories.csv')
orig_df = pd.read_csv('All orders - short custom description.csv', low_memory=False)
orig_df['date'] = pd.to_datetime(orig_df['date'])
orig_df.drop_duplicates(inplace=True)
orig_df.rename(columns={'quantity': 'Units', 'reference': 'SKU Reference', 'title': 'Product Name', 'price_inc': 'Revenue (£)', 'attribute_summary': 'Size'}, inplace=True)

today = datetime.datetime.now()
prev_year = today.year - 3
next_year = today.year
first_date = datetime.date(prev_year, 1, 1)
last_date = datetime.date(next_year, 12, 31)

d = st.date_input(
    "Select dates",
    (),
    first_date,
    last_date,
    format="DD.MM.YYYY",
    key=101,
)

d2 = ()

stand_options = 'None selected'

if(len(d) > 1):
    df = df[(df['date'] >= pd.to_datetime(d[0])) & (df['date'] <= pd.to_datetime(d[1]))]

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

        brand_sel_col, category_sel_col, sub_category_sel_col, colour_sel_col, size_sel_col, channel_sel_col = st.columns(6)

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
        
        channel_arr = df['Channel'].unique().tolist()
        channel_arr = np.sort(channel_arr).tolist()
        channel_arr = ['All channels'] + channel_arr
        channel_options = channel_sel_col.selectbox('Select a channel', options=channel_arr)
        if (channel_options == 'All channels'):
            df = df
        else:
            df = df[df['Channel'] == channel_options]

        d2 = st.date_input(
            "Comparison dates",
            (),
            first_date,
            last_date,
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

        stand_arr = ['None selected', 'Year-wise Sales', 'Mean Sales of 2021, 2022 and 2023', 'Sales Forecast for 2024',
                     'Google Ads Performance', 'SEO Backlink Analysis', 'Landing Page Engagement Rate']
        stand_options = st.selectbox('Standalone reports', options=stand_arr)
        if (stand_options == 'None selected'):
            df = df
    

    if ((stand_options == 'Year-wise Sales') & (filters_check == True)):
        year_options_arr = [2021, 2022, 2023]
        year_options = st.selectbox('Select Year', options=year_options_arr)
        df = orig_df[(orig_df['order_state'] == 'Order Dispatched') | (orig_df['order_state'] == 'Order Refunded')]
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df = df[df['year'] == year_options]

        st.markdown(f'<p class="big-font"><strong>Units</p>', unsafe_allow_html=True)
        df['month'] = df['date'].dt.month
        temp_df = df.groupby(['month', 'customs_description'])['Units'].sum().reset_index()
        newf = temp_df.pivot(index='customs_description', columns='month')
        newf.columns = newf.columns.droplevel(0)
        newf = newf.reindex(sorted(newf.columns), axis=1)
        month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        newf.columns = month_columns
        newf.loc['Total'] = newf.sum(numeric_only=True)
        newf.replace({np.NaN: 0}, inplace=True)
        newf = newf.astype(int)
        # newf.loc['Total', 'customs_description'] = 'Total'
        # newf.reset_index(drop=True, inplace=True)
        temp_df = newf.sort_values(by='Jan', ascending=False).reset_index()
        temp_df.rename(columns={'customs_description': 'Category'}, inplace=True)
        category_selection = dataframe_with_selections(temp_df, 21)

        if(category_selection['selected_rows_indices'] != []):
            selected_history = temp_df.loc[category_selection['selected_rows_indices'][0]]['Category']
            if selected_history != 'Total':
                some_df = df[df['customs_description'] == selected_history]
            else:
                some_df = df

            some_df['date'] = pd.to_datetime(some_df['date'])
            mean_df = some_df[some_df['date'].dt.year == year_options].set_index('date')
            mean_df = mean_df.resample('D')['Units'].sum().fillna(0).reset_index()

            mean_df = mean_df.groupby('date')['Units'].mean().reset_index().set_index('date')

            mean_df.reset_index(inplace=True)
            mean_df.rename(columns={'date': 'Date'}, inplace=True)
            mean_df = mean_df[['Date', 'Units']]
            mean_df['Units'] = mean_df['Units'].round(0).astype(int)
            mean_df['Date'] = pd.to_datetime(mean_df['Date'])


            # Create a selection that chooses the nearest point & selects based on x-value
            nearest = alt.selection_point(nearest=True, on='mouseover',
                                    fields=['Date'], empty=False)

            # The basic line
            line = alt.Chart(mean_df).mark_line().encode(
                x='Date',
                y='Units',
            ).interactive()

            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors = alt.Chart(mean_df).mark_point().encode(
                x='Date',
                y='Units',
                opacity=alt.value(0),
            ).add_params(
                nearest
            )

            # Draw points on the line, and highlight based on selection
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            # Draw a rule at the location of the selection
            rules = alt.Chart(mean_df).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )

            # Put the five layers into a chart and bind the data
            final = alt.layer(
                line, selectors, points, rules
            ).properties(
                width=600, height=300
            )
            st.altair_chart(final, use_container_width=True)


        st.markdown(f'<p class="big-font"><strong>Revenue (£)</p>', unsafe_allow_html=True)
        df['month'] = df['date'].dt.month
        temp_df_rev = df.groupby(['month', 'customs_description'])['Revenue (£)'].sum().reset_index()
        newf_rev = temp_df_rev.pivot(index='customs_description', columns='month')
        newf_rev.columns = newf_rev.columns.droplevel(0)
        newf_rev = newf_rev.reindex(sorted(newf_rev.columns), axis=1)
        month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        newf_rev.columns = month_columns
        newf_rev.loc['Total'] = newf_rev.sum(numeric_only=True)
        newf_rev.replace({np.NaN: 0}, inplace=True)
        newf_rev = newf_rev.astype(int)
        # newf.loc['Total', 'customs_description'] = 'Total'
        # newf.reset_index(drop=True, inplace=True)
        temp_df_rev = newf_rev.sort_values(by='Jan', ascending=False).reset_index()
        temp_df_rev.rename(columns={'customs_description': 'Category'}, inplace=True)
        category_selection_rev = dataframe_with_selections(temp_df_rev, 22)

        if(category_selection_rev['selected_rows_indices'] != []):
            selected_history_rev = temp_df_rev.loc[category_selection_rev['selected_rows_indices'][0]]['Category']
            if selected_history_rev != 'Total':
                some_df = df[df['customs_description'] == selected_history_rev]
            else:
                some_df = df

            some_df['date'] = pd.to_datetime(some_df['date'])
            mean_df = some_df[some_df['date'].dt.year == year_options].set_index('date')
            mean_df = mean_df.resample('D')['Revenue (£)'].sum().fillna(0).reset_index()

            mean_df = mean_df.groupby('date')['Revenue (£)'].mean().reset_index().set_index('date')

            mean_df.reset_index(inplace=True)
            mean_df.rename(columns={'date': 'Date'}, inplace=True)
            mean_df = mean_df[['Date', 'Revenue (£)']]
            mean_df['Revenue (£)'] = mean_df['Revenue (£)'].round(0).astype(int)
            mean_df['Date'] = pd.to_datetime(mean_df['Date'])


            # Create a selection that chooses the nearest point & selects based on x-value
            nearest = alt.selection_point(nearest=True, on='mouseover',
                                    fields=['Date'], empty=False)

            # The basic line
            line = alt.Chart(mean_df).mark_line().encode(
                x='Date',
                y='Revenue (£)',
            ).interactive()

            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors = alt.Chart(mean_df).mark_point().encode(
                x='Date',
                y='Revenue (£)',
                opacity=alt.value(0),
            ).add_params(
                nearest
            )

            # Draw points on the line, and highlight based on selection
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            # Draw a rule at the location of the selection
            rules = alt.Chart(mean_df).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )

            # Put the five layers into a chart and bind the data
            final = alt.layer(
                line, selectors, points, rules
            ).properties(
                width=600, height=300
            )
            st.altair_chart(final, use_container_width=True)            





    if ((stand_options == 'Mean Sales of 2021, 2022 and 2023') & (filters_check == True)):
        st.markdown(f'<p class="big-font"><strong>Mean Sales for 2021, 2022 and 2023</p>', unsafe_allow_html=True)
        df = orig_df[(orig_df['order_state'] == 'Order Dispatched') | (orig_df['order_state'] == 'Order Refunded')]
        df['date'] = pd.to_datetime(df['date'])
        l = [[2, 2021], [3, 2021], [1, 2022], [2, 2022], [3, 2022], [1, 2023], [2, 2023], [3, 2023], [4, 2021], [5, 2021], [6, 2021], [4, 2022], [5, 2022], [6, 2022],
             [4, 2023], [5, 2023], [6, 2023], [7, 2021], [8, 2021], [9, 2021], [7, 2022], [8, 2022], [9, 2022], [7, 2023], [8, 2023], [9, 2023], [10, 2021], [11, 2021],
             [12, 2021], [10, 2022], [11, 2022], [12, 2022], [10, 2023], [11, 2023], [12, 2023]]
        temp_df = df[(df['date'].dt.month == 1) & (df['date'].dt.year == 2021)].groupby('customs_description')['Units'].sum().reset_index().sort_values(by='Units', ascending=False).rename(columns={'Units': f'1-2021'})
        for i in l:
            temp_df = pd.merge(temp_df, df[(df['date'].dt.month == i[0]) & (df['date'].dt.year == i[1])].groupby('customs_description')['Units'].sum().reset_index().rename(columns={'Units': f'{i[0]}-{i[1]}'}), on='customs_description', how='outer')
        temp_df.replace({np.NaN: 0}, inplace=True)

        temp_df['Jan'] = temp_df[['1-2021', '1-2022', '1-2023']].mean(axis=1).round()
        temp_df['Feb'] = temp_df[['2-2021', '2-2022', '2-2023']].mean(axis=1).round()
        temp_df['Mar'] = temp_df[['3-2021', '3-2022', '3-2023']].mean(axis=1).round()
        temp_df['Apr'] = temp_df[['4-2021', '4-2022', '4-2023']].mean(axis=1).round()
        temp_df['May'] = temp_df[['5-2021', '5-2022', '5-2023']].mean(axis=1).round()
        temp_df['Jun'] = temp_df[['6-2021', '6-2022', '6-2023']].mean(axis=1).round()
        temp_df['Jul'] = temp_df[['7-2021', '7-2022', '7-2023']].mean(axis=1).round()
        temp_df['Aug'] = temp_df[['8-2021', '8-2022', '8-2023']].mean(axis=1).round()
        temp_df['Sep'] = temp_df[['9-2021', '9-2022', '9-2023']].mean(axis=1).round()
        temp_df['Oct'] = temp_df[['10-2021', '10-2022', '10-2023']].mean(axis=1).round()
        temp_df['Nov'] = temp_df[['11-2021', '11-2022', '11-2023']].mean(axis=1).round()
        temp_df['Dec'] = temp_df[['12-2021', '12-2022', '12-2023']].mean(axis=1).round()

        m = temp_df.select_dtypes(np.number)
        temp_df[m.columns] = m.round().astype('Int64')
        temp_df.rename(columns={'customs_description': 'Category'}, inplace=True)
        temp_df.loc['Total'] = temp_df.sum(numeric_only=True)
        temp_df.loc['Total', 'Category'] = 'Total'
        temp_df.reset_index(drop=True, inplace=True)
        temp_df = temp_df.sort_values(by='Jan', ascending=False).reset_index(drop=True)[['Category', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        history_selection = dataframe_with_selections(temp_df, 11)
        if(history_selection['selected_rows_indices'] != []):
            selected_history = temp_df.loc[history_selection['selected_rows_indices'][0]]['Category']
            if selected_history != 'Total':
                df = df[df['customs_description'] == selected_history]
            else:
                df = df

            df['date'] = pd.to_datetime(df['date'])
            df_2021 = df[df['date'].dt.year == 2021].set_index('date')
            df_2022 = df[df['date'].dt.year == 2022].set_index('date')
            df_2023 = df[df['date'].dt.year == 2023].set_index('date')

            df_2021 = df_2021.resample('D')['Units'].sum().fillna(0).reset_index()
            df_2022 = df_2022.resample('D')['Units'].sum().fillna(0).reset_index()
            df_2023 = df_2023.resample('D')['Units'].sum().fillna(0).reset_index()

            mean_df = pd.concat((df_2021, df_2022, df_2023))

            mean_df['date'] = mean_df['date'].astype(str)
            temp2_df = mean_df['date'].str.split('-', expand=True)
            temp2_df.columns = ['f_date', 's_date', 't_date']
            mean_df['date'] = temp2_df['s_date'] + '-' + temp2_df['t_date']

            mean_df = mean_df.groupby('date')['Units'].mean().reset_index().set_index('date')

            mean_df.reset_index(inplace=True)
            mean_df['date'] = '2024-' + mean_df['date']
            mean_df.rename(columns={'date': 'Date'}, inplace=True)
            mean_df = mean_df[['Date', 'Units']]
            mean_df['Units'] = mean_df['Units'].round(0).astype(int)
            mean_df['Date'] = pd.to_datetime(mean_df['Date'])


            # Create a selection that chooses the nearest point & selects based on x-value
            nearest = alt.selection_point(nearest=True, on='mouseover',
                                    fields=['Date'], empty=False)

            # The basic line
            line = alt.Chart(mean_df).mark_line().encode(
                x='Date',
                y='Units',
            ).interactive()

            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors = alt.Chart(mean_df).mark_point().encode(
                x='Date',
                y='Units',
                opacity=alt.value(0),
            ).add_params(
                nearest
            )

            # Draw points on the line, and highlight based on selection
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            # Draw a rule at the location of the selection
            rules = alt.Chart(mean_df).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )

            # Put the five layers into a chart and bind the data
            final = alt.layer(
                line, selectors, points, rules
            ).properties(
                width=600, height=300
            )
            st.altair_chart(final, use_container_width=True)




    elif ((stand_options == 'Sales Forecast for 2024') & (filters_check == True)):
        st.markdown(f'<p class="big-font"><strong>Sales Forecast for 2024</p>', unsafe_allow_html=True)
        categories = ['Total', 'Mens > Leather Jackets', 'Ladies > Leather Jackets', 'Gift & Accessories > Purses', 'Luggage > Holdall', 'Handbags > Ashwood',
              'Ladies > Ladies Jackets & Coats', 'Mens > Man Bags & Briefcases', 'Mens > Accessories', 'Ladies > Handbags',
              'Ladies > Purses', 'Ladies > Accessories']
        rev_categories = ['Revenue - ' + s for s in categories]
        
        dai_forecast_df = pd.read_csv('Daily category forecast [no features].csv')
        dai_forecast_df['Date'] = dai_forecast_df['Date'].astype(str)
        temp = dai_forecast_df['Date'].str.split('-', expand=True)
        temp.columns = ['Year', 'Month', 'Day']
        dai_forecast_df[['Year', 'Month', 'Day']] = temp[['Year', 'Month', 'Day']]

        mon_forecast_df = dai_forecast_df.groupby('Month').sum().drop(['Date', 'Year', 'Day'], axis=1)
        mon_forecast_df.reset_index(inplace=True)
        mon_forecast_df['Month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        l = []
        for i in categories:
            mon_forecast_df[i] += 1

        st.markdown(f'<p class="big-font"><strong>Units</p>', unsafe_allow_html=True)
        temp_categories = ['Month'] + categories
        mon_units_forecast_df = mon_forecast_df[temp_categories]
        temp_mon_units_forecast_df = mon_units_forecast_df.set_index('Month').T.reset_index()
        temp_mon_units_forecast_df.rename(columns={'index': 'Category'}, inplace=True)
        forecast_units_selection = dataframe_with_selections(temp_mon_units_forecast_df, 13)
        columns_units_list = mon_units_forecast_df.columns
        total_units_df = pd.DataFrame(columns=columns_units_list)
        total_units_df.loc['Total'] = mon_units_forecast_df.select_dtypes(np.number).sum()
        total_units_df['Month'] = 'Total'
        # st.dataframe(total_df, hide_index=True, use_container_width=True)

        if(forecast_units_selection['selected_rows_indices'] != []):
            selected_units_forecast = temp_mon_units_forecast_df.loc[forecast_units_selection['selected_rows_indices'][0]]['Category']

            dai_units_forecast_df = dai_forecast_df[['Date', selected_units_forecast]]
            dai_units_forecast_df['Date'] = pd.to_datetime(dai_units_forecast_df['Date'])

            # Create a selection that chooses the nearest point & selects based on x-value
            nearest = alt.selection_point(nearest=True, on='mouseover',
                                    fields=['Date'], empty=False)

            # The basic line
            line = alt.Chart(dai_units_forecast_df).mark_line().encode(
                x='Date',
                y=selected_units_forecast,
            ).interactive()

            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors = alt.Chart(dai_units_forecast_df).mark_point().encode(
                x='Date',
                y=selected_units_forecast,
                opacity=alt.value(0),
            ).add_params(
                nearest
            )

            # Draw points on the line, and highlight based on selection
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            # Draw a rule at the location of the selection
            rules = alt.Chart(dai_units_forecast_df).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )

            # Put the five layers into a chart and bind the data
            final = alt.layer(
                line, selectors, points, rules
            ).properties(
                width=600, height=300
            )
            st.altair_chart(final, use_container_width=True)


        st.markdown(f'<p class="big-font"><strong>Revenue (£)</p>', unsafe_allow_html=True)
        temp_rev_categories = ['Month'] + rev_categories
        mon_rev_forecast_df = mon_forecast_df[temp_rev_categories]
        mon_rev_forecast_df.columns = mon_rev_forecast_df.columns.str.replace('Revenue - ','')
        temp_mon_rev_forecast_df = mon_rev_forecast_df.set_index('Month').T.reset_index()
        temp_mon_rev_forecast_df.rename(columns={'index': 'Category'}, inplace=True)
        forecast_rev_selection = dataframe_with_selections(temp_mon_rev_forecast_df, 12)
        columns_rev_list = mon_rev_forecast_df.columns
        total_rev_df = pd.DataFrame(columns=columns_rev_list)
        total_rev_df.loc['Total'] = mon_rev_forecast_df.select_dtypes(np.number).sum()
        total_rev_df['Month'] = 'Total'
        # st.dataframe(total_df, hide_index=True, use_container_width=True)

        if(forecast_rev_selection['selected_rows_indices'] != []):
            selected_rev_forecast = temp_mon_rev_forecast_df.loc[forecast_rev_selection['selected_rows_indices'][0]]['Category']
            selected_rev_forecast = 'Revenue - ' + selected_rev_forecast

            dai_rev_forecast_df = dai_forecast_df[['Date', selected_rev_forecast]]
            dai_rev_forecast_df['Date'] = pd.to_datetime(dai_rev_forecast_df['Date'])

            # Create a selection that chooses the nearest point & selects based on x-value
            nearest_rev = alt.selection_point(nearest=True, on='mouseover',
                                    fields=['Date'], empty=False)

            # The basic line
            line_rev = alt.Chart(dai_rev_forecast_df).mark_line().encode(
                x='Date',
                y=selected_rev_forecast,
            ).interactive()

            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors_rev = alt.Chart(dai_rev_forecast_df).mark_point().encode(
                x='Date',
                y=selected_rev_forecast,
                opacity=alt.value(0),
            ).add_params(
                nearest_rev
            )

            # Draw points on the line, and highlight based on selection
            points_rev = line_rev.mark_point().encode(
                opacity=alt.condition(nearest_rev, alt.value(1), alt.value(0))
            )

            # Draw a rule at the location of the selection
            rules_rev = alt.Chart(dai_rev_forecast_df).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest_rev
            )

            # Put the five layers into a chart and bind the data
            final_rev = alt.layer(
                line_rev, selectors_rev, points_rev, rules_rev
            ).properties(
                width=600, height=300
            )
            st.altair_chart(final_rev, use_container_width=True)




    elif ((stand_options == 'Google Ads Performance') & (filters_check == True)):
        ads_df = pd.read_csv('2024_GoogleAdsCosts.csv', parse_dates=['date'])
        ads_df = ads_df[(ads_df['date'] >= pd.to_datetime(d[0])) & (ads_df['date'] <= pd.to_datetime(d[1]))]
        # disp_ads_df = ads_df.groupby('Campaign')[['Interactions', 'Clicks', 'Costs']].sum().reset_index()
        disp_ads_df = ads_df.groupby('Campaign')[['Clicks', 'Costs']].sum().reset_index()
        disp_ads_df['Costs'] = disp_ads_df['Costs'].astype(int)
        disp_ads_df['Clicks'] = disp_ads_df['Clicks'].replace(np.NaN, 0)
        disp_ads_df = disp_ads_df[disp_ads_df['Costs'] != 0]
        disp_ads_df['Clicks per pound'] = disp_ads_df['Clicks'] / disp_ads_df['Costs']
        disp_ads_df['Clicks per pound'] = disp_ads_df['Clicks per pound'].round(2)
        disp_ads_df['Percentage of Allocated Budget'] = disp_ads_df['Costs'] / disp_ads_df['Costs'].sum() * 100
        disp_ads_df['Percentage of Allocated Budget'] = disp_ads_df['Percentage of Allocated Budget'].round(2)
        disp_ads_df.rename(columns={'Costs': 'Costs (£)'}, inplace=True)
        disp_ads_df.sort_values(by='Clicks', ascending=False, inplace=True)
        disp_ads_df.reset_index(drop=True, inplace=True)
        st.markdown(f'<p class="big-font"><strong>Performance of All Campaigns from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="small-font"><strong>Total Budget:</strong> £{int(disp_ads_df["Costs (£)"].sum())}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="small-font"><strong>Total Clicks:</strong> {int(disp_ads_df["Clicks"].sum())}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="small-font"><strong>Clicks per pound:</strong> {np.round((disp_ads_df["Clicks"].sum() / disp_ads_df["Costs (£)"].sum()), 2)}</p>', unsafe_allow_html=True)
        campaign_selection = dataframe_with_selections(disp_ads_df, 13)
    
        
        if(campaign_selection['selected_rows_indices'] != []):
            selected_campaign = disp_ads_df.loc[campaign_selection['selected_rows_indices'][0]]['Campaign']
            
            graph_df = ads_df[ads_df['Campaign'] == selected_campaign]
            graph_df.rename(columns={'date': 'Date'}, inplace=True)
            graph_df['Date'] = pd.to_datetime(graph_df['Date'])
            graph_df['Clicks'] = graph_df['Clicks'].replace(np.NaN, 0)
            graph_df['Costs'] = graph_df['Costs'].astype(int)
            # graph_df['Interactions'] = graph_df['Interactions'].astype(int)

            graph_df = graph_df[['Date', 'Clicks', 'Costs']]
            data = graph_df.melt('Date')
            # line = alt.Chart(data).mark_line(point=True).encode(
            #     x='Date',
            #     y='value',
            #     color='variable'
            # ).interactive().properties(width=100, height=600)

            # Create a selection that chooses the nearest point & selects based on x-value
            nearest = alt.selection_point(nearest=True, on='mouseover',
                                    fields=['Date'], empty=False)

            # The basic line
            line = alt.Chart(data).mark_line().encode(
                x='Date',
                y='value',
                color='variable'
            ).interactive()

            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors = alt.Chart(data).mark_point().encode(
                x='Date',
                y='value',
                color='variable',
                opacity=alt.value(0),
            ).add_params(
                nearest
            )

            # Draw points on the line, and highlight based on selection
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            # Draw a rule at the location of the selection
            rules = alt.Chart(data).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )

            # Put the five layers into a chart and bind the data
            final = alt.layer(
                line, selectors, points, rules
            ).properties(
                width=600, height=300
            )

            # line1 = alt.Chart(graph_df, title=f'{selected_campaign} performance from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}').mark_line().encode(x='Date', y='Interactions').interactive()
            # line2 = alt.Chart(graph_df, title=f'{selected_campaign} performance from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}').mark_line().encode(x='Date', y='Costs').interactive()
            # line = alt.layer(line1, line2).resolve_scale(color='independent')

            st.write(f'{selected_campaign} Campaign Performance from {d[0].strftime("%d %b %Y")} to {d[1].strftime("%d %b %Y")}')
            st.altair_chart(final, use_container_width=True)

    elif ((stand_options == 'SEO Backlink Analysis') & (filters_check == True)):
        df_pages = pd.read_csv("Pages.csv")
        df_links = pd.read_csv("AlmostAllBacklinks.csv")

        type_arr = ['All backlinks', 'Category pages backlinks', 'Product pages backlinks']
        type_options = st.selectbox('Select a page type', options=type_arr)
        if type_options == 'All backlinks':
            df_links = df_links
        elif type_options == 'Category pages backlinks':
            df_links = df_links[df_links['Type'] == 'Category']
        elif type_options == 'Product pages backlinks':
            df_links = df_links[df_links['Type'] == 'Product']

        df = pd.merge(df_links, df_pages, on='Target page', how='left')
        df = df.dropna().reset_index(drop=True)

        stock_df = stock_df.groupby(['Product Name', 'Target page'])['Stock'].sum().reset_index()
        df = pd.merge(df, stock_df, on='Target page', how='left')

        if (type_options == 'Category pages backlinks'):
            chart = alt.Chart(df).mark_circle(size=60).encode(
                x='Incoming links',
                y='Google Search Position',
                color='Type',
                tooltip=['Target page', 'Incoming links', 'Google Search Position', 'Impressions', 'Clicks', 'CTR']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        
        else:
            chart = alt.Chart(df).mark_circle(size=60).encode(
                x='Incoming links',
                y='Google Search Position',
                color='Type',
                tooltip=['Target page', 'Incoming links', 'Google Search Position', 'Impressions', 'Clicks', 'CTR', 'Stock']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        product_df = df[df['Type'] == 'Product']
        product_df.drop(['Type'], axis=1, inplace=True)
        if type_options != 'Category pages backlinks':
            st.dataframe(product_df.sort_values(by='Stock').reset_index(drop=True), use_container_width=True)

        # chart2 = alt.Chart(df).mark_circle(size=60).encode(
        #     x='Clicks',
        #     y='Impressions',
        #     color='Type',
        #     tooltip=['Target page', 'Incoming links', 'Google Search Position', 'Impressions', 'Clicks', 'CTR']
        # ).interactive()
        # st.altair_chart(chart2, use_container_width=True)

    elif ((stand_options == 'Landing Page Engagement Rate') & (filters_check == True)):
        df_eng = pd.read_csv('EngagementBounceRate.csv', skiprows=9)
        df_eng.rename(columns={'Average engagement time per session': 'Average engagement time'}, inplace=True)
        df_eng['Average engagement time'] = df_eng['Average engagement time'].round(2)
        df_eng['Engagement rate'] = df_eng['Engagement rate'].round(2)
        df_eng['Bounce rate'] = df_eng['Bounce rate'].round(2)
        st.dataframe(df_eng, use_container_width=True)

    elif stand_options == 'None selected':
        df['Revenue (£)'] = df['Revenue (£)'].astype(float)
        refunded_df = df[df['order_state'] == 'Order Refunded']
        dispatched_df = df[(df['order_state'] == 'Order Dispatched') | (df['order_state'] == 'Order Refunded')]
        refunded_df.rename(columns={'Units': 'Units Refunded', 'Revenue (£)': 'Total Refund (£)'}, inplace=True)

        if (len(d2) == 2):
            orig_df['Product Name'] = orig_df['Product Name'].apply(str)
            orig_df['Product Name'] = orig_df['Product Name'].str.strip()
            orig_df['model'] = orig_df['model'].apply(str)
            orig_df['model'] = orig_df['model'].str.strip()
            orig_df['SKU Reference'] = orig_df['SKU Reference'].apply(str)
            orig_df['SKU Reference'] = orig_df['SKU Reference'].str.strip()

            df2 = orig_df[(orig_df['date'] >= pd.to_datetime(d2[0])) & (orig_df['date'] <= pd.to_datetime(d2[1]))]

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

            dispatched_df_temp = dispatched_df.copy(deep=False)
            refunded_df_temp = refunded_df.copy(deep=False)

            dispatched_df = pd.concat([dispatched_df, dispatched_df2], ignore_index=True)
            refunded_df = pd.concat([refunded_df, refunded_df2], ignore_index=True)


        if(d2 == ()):
            c1, c2, c3, c4 = st.columns(4)

            c1 = c1.container(border=True)
            c1.markdown(f'<p class="small-font">Total Revenue</p>', unsafe_allow_html=True)
            c1.markdown(f'<p class="big-font"><strong>£{int(np.round(dispatched_df["Revenue (£)"].sum(), 0)):,}</strong></p>', unsafe_allow_html=True)

            c2 = c2.container(border=True)
            c2.markdown(f'<p class="small-font">Units Sold</p>', unsafe_allow_html=True)
            c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong></p>', unsafe_allow_html=True)

            c3 = c3.container(border=True)
            c3.markdown(f'<p class="small-font">Total Refund</p>', unsafe_allow_html=True)
            c3.markdown(f'<p class="big-font"><strong>£{int(np.round(refunded_df["Total Refund (£)"].sum(), 0)):,}</strong></p>', unsafe_allow_html=True)

            c4 = c4.container(border=True)
            c4.markdown(f'<p class="small-font">Units Refunded</p>', unsafe_allow_html=True)
            c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong></p>', unsafe_allow_html=True)


        if (len(d2) == 2):
            c1, c2, c3, c4 = st.columns(4)

            p1 = percentage_change(np.round(dispatched_df_temp['Revenue (£)'].sum(), 2), np.round(dispatched_df2['Revenue (£)'].sum(), 2))
            c1 = c1.container(border=True)
            c1.markdown(f'<p class="small-font">Total Revenue</p>', unsafe_allow_html=True)
            if p1 > 0:
                c1.markdown(f'<p class="big-font"><strong>£{int(np.round(dispatched_df["Revenue (£)"].sum(), 0)):,}</strong><span style="color: green;"> (+{p1}%)</span></p>', unsafe_allow_html=True)
            elif p1 == 0:
                c1.markdown(f'<p class="big-font"><strong>£{int(np.round(dispatched_df["Revenue (£)"].sum(), 0)):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
            else:
                c1.markdown(f'<p class="big-font"><strong>£{int(np.round(dispatched_df["Revenue (£)"].sum(), 0)):,}</strong><span style="color: red;"> ({p1}%)</span></p>', unsafe_allow_html=True)

            p2 = percentage_change(dispatched_df_temp['Units'].sum(), dispatched_df2['Units'].sum())
            c2 = c2.container(border=True)
            c2.markdown(f'<p class="small-font">Units Sold</p>', unsafe_allow_html=True)
            if p2 > 0:
                c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong><span style="color: green;"> (+{p2}%)</span></p>', unsafe_allow_html=True)
            elif p2 == 0:
                c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
            else:
                c2.markdown(f'<p class="big-font"><strong>{(dispatched_df["Units"].sum()):,}</strong><span style="color: red;"> ({p2}%)</span></p>', unsafe_allow_html=True)

            p3 = percentage_change((np.round(refunded_df_temp['Total Refund (£)'].sum(), 2)), (np.round(refunded_df2['Total Refund (£)'].sum(), 2)))
            c3 = c3.container(border=True)
            c3.markdown(f'<p class="small-font">Total Refund</p>', unsafe_allow_html=True)
            if p3 > 0:
                c3.markdown(f'<p class="big-font"><strong>£{int(np.round(refunded_df["Total Refund (£)"].sum(), 0)):,}</strong><span style="color: red;"> (+{p3}%)</span></p>', unsafe_allow_html=True)
            elif p3 == 0:
                c3.markdown(f'<p class="big-font"><strong>£{int(np.round(refunded_df["Total Refund (£)"].sum(), 0)):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
            else:
                c3.markdown(f'<p class="big-font"><strong>£{int(np.round(refunded_df["Total Refund (£)"].sum(), 0)):,}</strong><span style="color: green;"> ({p3}%)</span></p>', unsafe_allow_html=True)

            p4 = percentage_change(refunded_df_temp['Units Refunded'].sum(), refunded_df2['Units Refunded'].sum())
            c4 = c4.container(border=True)
            c4.markdown(f'<p class="small-font">Units Refunded</p>', unsafe_allow_html=True)
            if p4 > 0:
                c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong><span style="color: red;"> (+{p4}%)</span></p>', unsafe_allow_html=True)
            elif p4 == 0:
                c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong><span style="color: black;"> (0%)</span></p>', unsafe_allow_html=True)
            else:
                c4.markdown(f'<p class="big-font"><strong>{(refunded_df["Units Refunded"].sum()):,}</strong><span style="color: green;"> ({p4}%)</span></p>', unsafe_allow_html=True)

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
                        dispatched_product_three_cat_df = product_condense_dataframe(dispatched_df, refunded_df)

                        product_stock_df = stock_df.groupby('Product Name')['Stock'].sum()
                        dispatched_product_three_cat_df = pd.merge(dispatched_product_three_cat_df, product_stock_df, how="left", on="Product Name")
                        dispatched_product_three_cat_df['Stock'] = dispatched_product_three_cat_df['Stock'].replace(np.NaN, 0)

                        # dispatched_product_three_cat_df['index'] = range(1, len(dispatched_product_three_cat_df) + 1)
                        # column_to_move = dispatched_product_three_cat_df.pop("index")
                        # dispatched_product_three_cat_df.insert(0, "index", column_to_move)
                        
                        selection4 = dataframe_with_selections(dispatched_product_three_cat_df, 4)
                        if(selection4['selected_rows_indices'] != []):
                            selected_prod = dispatched_product_three_cat_df.loc[selection4['selected_rows_indices'][0]]['Product Name']
                            
                            dispatched_df = dispatched_df[dispatched_df['Product Name'] == selected_prod]
                            refunded_df = refunded_df[refunded_df['Product Name'] == selected_prod]
                            stock_df = stock_df[stock_df['Product Name'] == selected_prod]

                            dispatched_sku_three_cat_df = sku_condense_dataframe(dispatched_df, refunded_df)

                            sku_stock_df = stock_df.groupby('Size')['Stock'].sum()
                            dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, sku_stock_df, how="outer", on="Size")
                            dispatched_sku_three_cat_df['Stock'] = dispatched_sku_three_cat_df['Stock'].replace(np.NaN, 0)

                            display_sku(selected_prod, d, d2, dispatched_df, dispatched_sku_three_cat_df)

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

                            dispatched_product_three_cat_df = product_condense_dataframe(dispatched_df, refunded_df)

                            product_stock_df = stock_df.groupby('Product Name')['Stock'].sum()
                            dispatched_product_three_cat_df = pd.merge(dispatched_product_three_cat_df, product_stock_df, how="left", on="Product Name")
                            dispatched_product_three_cat_df['Stock'] = dispatched_product_three_cat_df['Stock'].replace(np.NaN, 0)

                            # dispatched_product_three_cat_df['index'] = range(1, len(dispatched_product_three_cat_df) + 1)
                            # column_to_move = dispatched_product_three_cat_df.pop("index")
                            # dispatched_product_three_cat_df.insert(0, "index", column_to_move)

                            selection5 = dataframe_with_selections(dispatched_product_three_cat_df, 5)
                            if(selection5['selected_rows_indices'] != []):
                                selected_prod = dispatched_product_three_cat_df.loc[selection5['selected_rows_indices'][0]]['Product Name']
                                
                                dispatched_df = dispatched_df[dispatched_df['Product Name'] == selected_prod]
                                refunded_df = refunded_df[refunded_df['Product Name'] == selected_prod]
                                stock_df = stock_df[stock_df['Product Name'] == selected_prod]

                                dispatched_sku_three_cat_df = sku_condense_dataframe(dispatched_df, refunded_df)

                                sku_stock_df = stock_df.groupby('Size')['Stock'].sum()
                                dispatched_sku_three_cat_df = pd.merge(dispatched_sku_three_cat_df, sku_stock_df, how="outer", on="Size")
                                dispatched_sku_three_cat_df['Stock'] = dispatched_sku_three_cat_df['Stock'].replace(np.NaN, 0)

                                display_sku(selected_prod, d, d2, dispatched_df, dispatched_sku_three_cat_df)

            else:
                dispatched_product_two_cat_df = product_condense_dataframe(dispatched_df, refunded_df)

                product_stock_df = stock_df.groupby('Product Name')['Stock'].sum()
                dispatched_product_two_cat_df = pd.merge(dispatched_product_two_cat_df, product_stock_df, how="left", on="Product Name")
                dispatched_product_two_cat_df['Stock'] = dispatched_product_two_cat_df['Stock'].replace(np.NaN, 0)

                # dispatched_product_two_cat_df['index'] = range(1, len(dispatched_product_two_cat_df) + 1)
                # column_to_move = dispatched_product_two_cat_df.pop("index")
                # dispatched_product_two_cat_df.insert(0, "index", column_to_move)

                selection4 = dataframe_with_selections(dispatched_product_two_cat_df, 4)
                if(selection4['selected_rows_indices'] != []):
                    selected_prod = dispatched_product_two_cat_df.loc[selection4['selected_rows_indices'][0]]['Product Name']

                    dispatched_df = dispatched_df[dispatched_df['Product Name'] == selected_prod]
                    refunded_df = refunded_df[refunded_df['Product Name'] == selected_prod]
                    stock_df = stock_df[stock_df['Product Name'] == selected_prod]

                    dispatched_sku_two_cat_df = sku_condense_dataframe(dispatched_df, refunded_df)

                    product_stock_df = stock_df.groupby('Size')['Stock'].sum()
                    dispatched_sku_two_cat_df = pd.merge(dispatched_sku_two_cat_df, product_stock_df, how="outer", on="Size")
                    dispatched_sku_two_cat_df['Stock'] = dispatched_sku_two_cat_df['Stock'].replace(np.NaN, 0)

                    display_sku(selected_prod, d, d2, dispatched_df, dispatched_sku_two_cat_df)