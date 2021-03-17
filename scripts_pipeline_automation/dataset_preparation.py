#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import pickle

import pandas as pd


def parser(x):
    return datetime.strptime(str(x), '%m/%d/%Y')


def write_pickle_serialized_file(filename, content):
    outfile = open(filename, 'wb')
    pickle.dump(content, outfile)
    outfile.close()

def read_vertical_dataset(filename):
    df_dataset_vertical = pd.read_csv(filename)
    df_dataset_vertical = df_dataset_vertical[['SITE','DATES','OCCUPANCY_COUNT']]
    return df_dataset_vertical


def write_df_grouped_dates_file(filename, df_dataset_vertical):
    df_grouped_dates = df_dataset_vertical.groupby(['DATES'], as_index=False).sum()
    df_grouped_dates.to_csv(filename, index=False)
    print('Saving {} ...'.format(filename))
    return df_grouped_dates


def write_series_grouped_dates_file(filename, period):
    series_grouped_dates = pd.read_csv(
        filename, 
        header=0, 
        index_col=0, 
        parse_dates=True,
        squeeze=True,
        date_parser=parser
    )
    series_grouped_dates.index = series_grouped_dates.index.to_period(period)
    return series_grouped_dates


def write_pandas_dataframe_groupby_site_serialized_object(filename, df_dataset_vertical):
    pandas_dataframe_groupby_site = df_dataset_vertical.groupby(['SITE'], as_index=False)
    write_pickle_serialized_file(filename, pandas_dataframe_groupby_site)
    print('Saving {} ...'.format(filename))
    return pandas_dataframe_groupby_site


def write_sites_names_serialized_list(filename, pandas_dataframe_groupby_site):
    sites_names = list(pandas_dataframe_groupby_site.groups.keys())
    write_pickle_serialized_file(filename, sites_names)
    print('Saving {} ...'.format(filename))
    return sites_names


def generate_dataset_files(path, dataset_name, data_period):
    df_dataset_vertical = read_vertical_dataset('{}{}_vertical.csv'.format(path, dataset_name))

    df_grouped_dates = write_df_grouped_dates_file(
        '{}{}_vertical_grouped_dates.csv'.format(path, dataset_name),
        df_dataset_vertical
    )

    series_grouped_dates = write_series_grouped_dates_file(
        '{}{}_vertical_grouped_dates.csv'.format(path, dataset_name),
        period=data_period
    )

    pandas_dataframe_groupby_site = write_pandas_dataframe_groupby_site_serialized_object(
        '{}{}_pandas_dataframe_groupby_site.pkl'.format(path, dataset_name),
        df_dataset_vertical
    )

    sites_names = write_sites_names_serialized_list(
        '{}{}_sites_names_list.pkl'.format(path, dataset_name),
        pandas_dataframe_groupby_site
    )
    print('\nAll files have been saved succesfully!')
    
    return df_grouped_dates, series_grouped_dates, pandas_dataframe_groupby_site, sites_names