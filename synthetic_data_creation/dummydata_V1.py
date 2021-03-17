import pandas as pd
import numpy as np
import random as rd
from random import randint
import datetime
from datetime import datetime, timedelta
#from datetime import date, time

#path = 'C:\Users\USUARIO\Documents\Github\data'
path = './'


def DateTimeRangeFunc(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    DateRange = [datetime.fromordinal(i) for i in range(
        start_date.toordinal(), end_date.toordinal())]
    DateTimeRange = [str(fecha+timedelta(hours=hr))
                     for fecha in DateRange for hr in range(1, 24)]
    return DateTimeRange


df_occupancy = pd.read_excel(path+'boa.xlsx')
SitesRange = df_occupancy.name.unique()
SitesRange


def OccupancyRangeFunc(lenght):
    OccupancyRange = np.random.poisson(randint(1, 10), lenght)
    return OccupancyRange


def GenerarDBFicticia(SitesRange, start_date_str, end_date_str, direcction='V'):
    list_df = []
    for item in SitesRange:
        DateTimeRange = DateTimeRangeFunc(start_date_str, end_date_str)
        # OccupancyRange = OccupancyRangeFunc(len(DateTimeRange))
        OccupancyRange = []
        for i in range(149):
            #ST_occup generated each time instead of copying values sequencially
            ST_occup_aux = Generate_ST_occup()
            OccupancyRange.extend(ST_occup_aux)
        #Last ST_occup list
        ST_occup = Generate_ST_occup()
        OccupancyRange.extend(ST_occup[0:153])
        # OccupancyRange = OccupancyRangeDate(ST_occup,'2018-01-01','2020-12-31')
        df_aux = pd.DataFrame(
            {'DATE': DateTimeRange, 'OCCUPANCY_COUNT': OccupancyRange})
        df_aux['SITE'] = item
        list_df.append(df_aux)

    df_registros = pd.concat(list_df)
    df_registros = df_registros[['SITE', 'DATE', 'OCCUPANCY_COUNT']]

    if direcction == 'H':
        df_registros = df_registros.pivot(
            index='SITE', columns='DATE', values='OCCUPANCY_COUNT')
        df_registros.reset_index(inplace=True)

    return df_registros

# TODO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  NUEVAS FUNCIONES
# week={0:[],1:[],2:[],3:[],4:[],5:[]}
'''
n=100
x=np.arange(n)
y_=np.random.uniform(-1,1,[n])

mu=0
sigma=0.01
e= np.random.normal(mu, sigma, n)
#stationary series
y=y_+e

plt.plot(x,y)
plt.show()
yt=yt−1+εt

The nonstationary is also a very broad category: anything with changing mean or variance. 
You can change mean and variance deterministically or stochastically. 
The simplest process could be: yt=yt−1εt or even yt=yt−1+εt, which you can generate by simple recursion or a loop
'''
def GenerateDay(week):
    e= rd.randint(1,5)
    num_h = 24
    for day in week:
        d_low = int(week[day][0]) - 1 + e
        d_high = int(week[day][1]) - 1 + e
        print("LI: "+str(d_low)+" LS: "+str(d_high)+" Day: "+str(day))
        week[day] = list(np.random.poisson(randint(d_low, d_high), num_h))
    return week

# session_hours =[9,10,11,12]
# hours_distribution = [(25,45),(12,16),(18,15),(12,11)]


# ooc_week es un diccionario de claves dia (0:5) y valores occupacy(0:23)
# ooc_week es un diccionario de claves dia (0:5) y valores occupacy(0:23)
def ModifyHourDay(occ_week, session_hours, hours_distribution, day=0):
    e= rd.randint(1,5)
    i = 0
    for h in session_hours:
        h_low = int(hours_distribution[i][0]) - 1 + e
        h_high = int(hours_distribution[i][1]) - 1 + e
        # print("LI: "+str(h_low)+" LS: "+str(h_high)+" hour: "+str(h))
        occ_week[day][h] = int(np.random.poisson(randint(min([h_low, h_high]), max([h_low, h_high]))))
        i += 1
    return occ_week

# TODO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  TEST NUEVAS FUNCIONES

def Generate_ST_occup():
    #Functions parameters
    weekday_distribution_variation, session_hours, hours_distribution = Generate_ST_occup_parameters()

    #!First function
    weekday = GenerateDay(weekday_distribution_variation)
    dfweek = pd.DataFrame(weekday)
 
    #! Second function
    week_mod_h = ModifyHourDay(weekday,session_hours,hours_distribution)
    df_week_mod_h = pd.DataFrame(week_mod_h)
    df_week_mod_h = pd.concat([df_week_mod_h, pd.DataFrame({'5':[0]*24, '6':[0]*24})], axis=1)
    df_week_mod_h['hours'] = range(1,25)

    df_week_mod_h = df_week_mod_h.melt(id_vars=['hours'], var_name='date', value_name='occupancy')
    ST_occup = df_week_mod_h['occupancy'].to_list()

    return ST_occup

def Generate_ST_occup_parameters():
    #First function parameters
    weekday_distribution_variation = {
        0: (6, 11),
        1: (4, 9),
        2: (3, 9),
        3: (5, 10),
        4: (0, 2)
    }

    #Second function parameters
    session_hours = [9,10,11,12]
    hours_distribution = [(25,45),(12,16),(18,15),(12,11)] 

    return (weekday_distribution_variation, session_hours, hours_distribution)


'''
#!First function
weekday_distribution_variation = {
    0: (6, 11),
    1: (4, 9),
    2: (3, 9),
    3: (5, 10),
    4: (0, 2)
}

weekday = GenerateDay(weekday_distribution_variation) #GeneraDate to GenerateDate
dfweek = pd.DataFrame(weekday)
# dfweek

#! Second function
session_hours = [9,10,11,12]
hours_distribution = [(25,45),(12,16),(18,15),(12,11)]
week_mod_h=ModifyHourDay(weekday,session_hours,hours_distribution) #ModificaHourDay to ModifyHourDay
df_week_mod_h=pd.DataFrame(week_mod_h)
df_week_mod_h = pd.concat([df_week_mod_h, pd.DataFrame({'5':[0]*24, '6':[0]*24})], axis=1)
df_week_mod_h['hours'] = range(1,25)

df_week_mod_h = df_week_mod_h.melt(id_vars=['hours'], var_name='date', value_name='occupancy')
ST_occup = df_week_mod_h['occupancy'].to_list()
'''

# CREAR BASE DE REGISTROS
np.random.seed(1234)
df1 = GenerarDBFicticia(SitesRange, '2018-01-01', '2020-12-31', direcction='V')
df1.to_csv(path+'BaseVertical.csv')
df2 = GenerarDBFicticia(SitesRange, '2018-01-01', '2020-12-31', direcction='H')
df2.to_csv(path+'BaseHorizontal.csv')
    
