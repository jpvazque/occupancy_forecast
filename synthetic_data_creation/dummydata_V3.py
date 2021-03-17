# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from random import sample, randint, uniform

import pandas as pd
import numpy as np
from numpy.random import poisson, uniform, seed


PATH = 'C:/Users/59399/OneDrive/Documents/TRABAJO/DianaMosquera/Occupancy_Forecasting/Data_scripts_V2/BDForecasting_V2/'


def date_time_range_func(start_date_str, end_date_str, string=True):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    DateRange = [datetime.fromordinal(i) for i in range(
        start_date.toordinal(), (end_date.toordinal()+1))]
    
    if string:
        DateTimeRange = [str(date+timedelta(hours=hr))
                         for date in DateRange for hr in range(1, 24)]
    else:
        DateTimeRange = [date+timedelta(hours=hr)
                         for date in DateRange for hr in range(1, 24)]
    return DateTimeRange


def occupancy_range_func(lenght):
    OccupancyRange = poisson(randint(1, 10), lenght)
    return OccupancyRange 


def fill_random_values_by_date_section(df_dates, sessions_date_section, date_section):
    for date_section_id in sessions_date_section:
        d_low = int(sessions_date_section[date_section_id][0])
        d_high = int(sessions_date_section[date_section_id][1])
        length = df_dates.loc[df_dates[date_section]==date_section_id, 'OCCUPANCY_COUNT'].shape[0]
        df_dates.loc[df_dates[date_section]==date_section_id, 'OCCUPANCY_COUNT'] = poisson(randint(d_low, d_high), length)
    
    return df_dates


def fill_date_section(df_dates, date_section_rates, date_section):
    date_low_rates = date_section_rates[0]
    date_high_rates = date_section_rates[1]
    
    if date_low_rates:       
        for low_date_id in date_low_rates:
            low = uniform(0.85,  0.90)
            high = uniform(0.91, 0.95)
            date_low_occ = df_dates.loc[df_dates[date_section]==low_date_id, 'OCCUPANCY_COUNT']
            length = date_low_occ.shape[0]
            new_sessions_date_section = uniform(low, high, length)
            df_dates.loc[df_dates[date_section]==low_date_id, 'OCCUPANCY_COUNT'] = (date_low_occ * new_sessions_date_section).astype(int)
         
    if date_high_rates:
        for high_date_id in date_high_rates:
            low = uniform(1.15, 1.25)
            high = uniform(1.26, 1.35)
            date_high_occ = df_dates.loc[df_dates[date_section]==high_date_id, 'OCCUPANCY_COUNT']
            length = date_high_occ.shape[0]
            new_sessions_date_section = uniform(low, high, length)
            df_dates.loc[df_dates[date_section]==high_date_id, 'OCCUPANCY_COUNT'] = (date_high_occ * new_sessions_date_section).astype(int)

    return df_dates


def fill_values_by_hour_distribution(df_dates, sessions_hour):
    for hour_id in sessions_hour:
        d_low = int(sessions_hour[hour_id][0])
        d_high = int(sessions_hour[hour_id][1])
        length = df_dates.loc[df_dates['HOUR']==hour_id, 'OCCUPANCY_COUNT'].shape[0]
        new_sessions_hour = uniform(d_low, d_high, length)
        hour_occ = df_dates.loc[df_dates['HOUR']==hour_id, 'OCCUPANCY_COUNT']
        df_dates.loc[df_dates['HOUR']==hour_id, 'OCCUPANCY_COUNT'] = (hour_occ * new_sessions_hour).astype(int)

    return df_dates


def generate_sample_months(sample_number):
    months = set(range(1,13))
    months_highest_rates = set(sample(months, k=sample_number))
    months -= months_highest_rates
    months_lowest_rates = sample(months, k=sample_number)

    return (months_lowest_rates, list(months_highest_rates))


def generate_sample_weekdays(sample_number):
    days = set(range(0,7))
    days_highest_rates = set(sample(days, k=sample_number))
    days -= days_highest_rates
    days_lowest_rates = sample(days, k=sample_number)

    return (days_lowest_rates, list(days_highest_rates))


def generate_initial_registers(start_date_str, end_date_str):
    list_df = []
    for item in SitesRange:
        DateTimeRange  = date_time_range_func(start_date_str, end_date_str, string=False)
        OccupancyRange = occupancy_range_func(len(DateTimeRange))
        df_aux = pd.DataFrame({'DATES':DateTimeRange, 'OCCUPANCY_COUNT':OccupancyRange})
        df_aux['SITE'] = item
        list_df.append(df_aux)
    
    df_registros = pd.concat(list_df)
    df_registros = df_registros[['SITE','DATES', 'OCCUPANCY_COUNT']]
    
    return df_registros


def generate_dummy_DB(SitesRange, start_date_str, end_date_str, direcction='V'):
    df_registros = generate_initial_registers(start_date_str, end_date_str)
    
    if direcction == 'H':
        df_registros = df_registros.pivot(index='SITE', columns='DATE', values='OCCUPANCY_COUNT')
        df_registros.reset_index(inplace=True)
    
    return df_registros 


def generate_dummy_DB_V2(SitesRange, start_date_str, end_date_str, direcction='V', 
                         set_by_parameters = False, sessions_month=None, sessions_day=None, sessions_hour=None):
    df_registros = generate_initial_registers(start_date_str, end_date_str)
    
    if set_by_parameters:
        df_registros['MONTH']   = df_registros['DATES'].dt.month
        df_registros['WEEKDAY'] = df_registros['DATES'].dt.weekday
        df_registros['HOUR']    = df_registros['DATES'].dt.hour
        
        if sessions_month is not None:
            df_registros = fill_random_values_by_date_section(df_registros, sessions_month, 'MONTH') 
        if sessions_day is not None:
            df_registros = fill_random_values_by_date_section(df_registros, sessions_day, 'WEEKDAY')
        if sessions_hour is not None:
            df_registros = fill_random_values_by_date_section(df_registros, sessions_hour, 'HOUR')
    
    df_registros.drop(columns=['MONTH', 'WEEKDAY', 'HOUR'], inplace=True)
    
    if direcction == 'H':
        df_registros = df_registros.pivot(index='SITE', columns='DATES', values='OCCUPANCY_COUNT')
        df_registros.reset_index(inplace=True)
    
    return df_registros 


def generate_dummy_DB_V3(SitesRange, start_date_str, end_date_str, direcction='V', 
                         set_by_parameters = False, month_rates=None, day_rates=None, sessions_hour=None):
    df_registros = generate_initial_registers(start_date_str, end_date_str)
    
    if set_by_parameters:
        df_registros['MONTH']   = df_registros['DATES'].dt.month
        df_registros['WEEKDAY'] = df_registros['DATES'].dt.weekday
        df_registros['HOUR']    = df_registros['DATES'].dt.hour
        
        df_registros = fill_values_by_hour_distribution(df_registros, sessions_hour)
        
        if day_rates is not None:
            df_registros = fill_date_section(df_registros, day_rates, 'WEEKDAY')
        if month_rates is not None:
            df_registros = fill_date_section(df_registros, month_rates, 'MONTH')
    
    df_registros.drop(columns=['MONTH', 'WEEKDAY','HOUR'], inplace=True)
    
    if direcction == 'H':
        df_registros = df_registros.pivot(index='SITE', columns='DATES', values='OCCUPANCY_COUNT')
        df_registros.reset_index(inplace=True)
    
    return df_registros


df_occupancy = pd.read_excel(PATH+'boa.xlsx')

SitesRange = df_occupancy.name.unique()


months_highest_rates = [2,6,9]

months_lowest_rates = [1,12]

months_rates = (months_lowest_rates, months_highest_rates)

days_highest_rates = []

days_lowest_rates = [5, 6]

days_rates = (days_lowest_rates, days_highest_rates)

sessions_hour = {
    1 : (0.10, 0.19),
    2 : (0.10, 0.17),
    3 : (0.10, 0.18),
    4 : (0.10, 0.19),
    5 : (0.15, 0.20),
    6 : (0.20, 0.30),
    7 : (0.40, 0.50),
    8 : (0.50, 0.60),
    9 : (0.70, 0.80),
    10 : (0.71, 0.77),
    11 : (0.73, 0.79),
    12 : (0.50, 0.60),
    13 : (0.55, 0.65),
    14 : (0.75, 0.80),
    15 : (0.82, 0.85),
    16 : (0.80, 0.86),
    17 : (0.81, 0.84),
    18 : (0.80, 0.85),
    19 : (0.55, 0.60),
    20 : (0.50, 0.55),
    21 : (0.40, 0.50),
    22 : (0.30, 0.40),
    23 : (0.20, 0.30),
    24 : (0.10, 0.20)
    }


seed(1234) 
df_vertical = generate_dummy_DB_V3(SitesRange, '2018-01-01', '2020-12-31', direcction='V',
                           set_by_parameters = True, 
                           month_rates=months_rates,
                           day_rates=days_rates,
                           sessions_hour=sessions_hour)
df_vertical.to_csv(PATH+'BaseVertical_V4.csv')

df_horizontal = generate_dummy_DB_V3(SitesRange, '2018-01-01', '2020-12-31', direcction='H',
                           set_by_parameters = True, 
                           month_rates=months_rates,
                           day_rates=days_rates,
                           sessions_hour=sessions_hour)
df_horizontal.to_csv(PATH+'BaseHorizontal_V4.csv')