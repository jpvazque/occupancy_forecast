#!/usr/bin/env python
# coding: utf-8

import dataset_preparation
from  imports_models import *
from utils import *


def reset_model_info_dictionaries():
    global groups_model_info
    global exec_times_info
    
    groups_model_info = {
        'timeseries':[],
        'tuning_results':[],
        'models':[],
        'cross_validation_results':[],
        'forecasts':[]
    }
    exec_times_info = {
        'tuning_exec_times':[],
        'training_exec_times':[],
        'cross_validation_exec_times':[],
        'forecast_exec_times':[]
    }
    
    
def save_groups_model_info(
    site_df_grouped_dates=None, 
    site_tuning_results=None, 
    prophet_instance=None, 
    site_cross_validation_results=None, 
    site_forecast=None,
    tuning=True,
    cross_validation=True
):    
    groups_model_info['timeseries'].append(site_df_grouped_dates)
    groups_model_info['models'].append(prophet_instance)
    groups_model_info['forecasts'].append(site_forecast)
    
    if tuning:
        groups_model_info['tuning_results'].append(site_tuning_results)   
    if cross_validation:
        groups_model_info['cross_validation_results'].append(site_cross_validation_results)
        
    return groups_model_info
        

def save_model_exec_times_info(
    tuning_exec_time=None, 
    training_exec_time=None, 
    cross_validation_exec_time=None, 
    forecast_exec_time=None,
    tuning=True,
    cross_validation=True
):    
    exec_times_info['training_exec_times'].append(training_exec_time)
    exec_times_info['forecast_exec_times'].append(forecast_exec_time)
    
    if tuning:
        exec_times_info['tuning_exec_times'].append(tuning_exec_time)    
    if cross_validation:
        exec_times_info['cross_validation_exec_times'].append(cross_validation_exec_time)
    
    return exec_times_info


def write_model_info_serialized_file(filename):
    file = open(filename, 'wb')
    pickle.dump((save_groups_model_info, save_model_exec_times_info, overall_exec_time), file)
    file.close()


def generate_fbprophet_forecasting_model(
    all_sites=False, 
    sites=None, 
    site_index=0,
    initial=None,
    horizon=None
):    
    reset_model_info_dictionaries()
    
    if all_sites:
        print('Trainig and forecasting multiple time series groups...\n')
    else:
        print('Training and forecasting %s ...\n'%sites_names[site_index])

    start_time_overall = time.time()
    for site in tqdm(sites):
        print('\033[1m' + 'Group %s'%(site) + '\033[0m')
        site_complete_data = pandas_dataframe_groupby_site.get_group(site)
        site_df_grouped_dates = prepare_timeserie_df(site_complete_data)
        rows_len = site_df_grouped_dates.shape[0]
        training_size = 0.7
        site_df_data_train = site_df_grouped_dates[:int(rows_len*training_size)]
        site_df_data_test = site_df_grouped_dates[int(rows_len*training_size):]
        
        
        print('Tuning...')
        start_time_tuning = time.time()
        site_tuning_results = hyperparameter_tuning(
            site_df_data_train, 
            site, 
            initial=initial,
            horizon=horizon
        )
        tuning_exec_time = time.time() - start_time_tuning
        print("--- Tuning ended at %s seconds ---" % (tuning_exec_time))
        
        
        print('Training model...')
        start_time_training = time.time()
        prophet_instance = Prophet(
                                yearly_seasonality=True, 
                                weekly_seasonality=True,
                                seasonality_prior_scale=site_tuning_results[0],
                                changepoint_prior_scale=site_tuning_results[1],
                                interval_width=0.95
                            )
        prophet_instance.add_seasonality(name='monthly', 
                                        period=30.5, 
                                        fourier_order=5, 
                                        prior_scale=0.02)

        prophet_instance.fit(site_df_data_train)
        training_exec_time = time.time() - start_time_training
        print("--- Training ended at %s seconds ---" % (training_exec_time))
        
        
        print('Cross validation...')
        start_time_cross_validation = time.time()
        site_cross_validation_results = cross_validation(
                                            prophet_instance, 
                                            initial=initial, 
                                            horizon=horizon
        )
        cross_validation_exec_time = time.time() - start_time_cross_validation
        print("--- Cross validation ended at %s seconds ---" % (cross_validation_exec_time))
        
        
        print('Forecasting...')
        start_time_forecast = time.time()
        site_forecast = prophet_instance.predict(site_df_data_test)
        forecast_exec_time = time.time() - start_time_forecast
        print("--- Forecast ended at %s seconds ---\n" % (forecast_exec_time))
        
        save_groups_model_info(
            site_df_grouped_dates, 
            site_tuning_results, 
            prophet_instance, 
            site_cross_validation_results, 
            site_forecast,
            tuning=True,
            cross_validation=True
        )
        save_model_exec_times_info(
            tuning_exec_time, 
            training_exec_time, 
            cross_validation_exec_time, 
            forecast_exec_time,
            tuning=True,
            cross_validation=True
        )
        
    overall_exec_time = time.time() - start_time_overall
    print("Multiple Training and Forecasting ended at--- %s seconds ---" % (overall_exec_time))
    write_model_info_serialized_file('{}{}_prophet_model_info.pkl'.format(PATH, DATASET_NAME))


def generate_sarima_forecasting_model(
    all_sites=False, 
    sites=None, 
    site_index=0
):
    print('\nSARIMA MODEL\n')
    reset_model_info_dictionaries()
    
    if all_sites:
        print('Trainig and forecasting multiple time series groups...\n')
    else:
        print('Training and forecasting %s ...\n'%sites_names[site_index])

    start_time_overall = time.time()
    for site in tqdm(sites):
        print('\033[1m' + 'Group %s'%(site) + '\033[0m')
        site_df = pandas_dataframe_groupby_site.get_group(site)
        site_series_data = prepare_timeserie_df(
            site_df, 
            pd_series=True, 
            series_index=series_grouped_dates.index
        )
        rows_len = site_series_data.shape[0]
        training_size = 0.7
        site_series_train = site_series_data[:int(rows_len*training_size)]
        site_series_test = site_series_data[int(rows_len*training_size):]
        
        
        print('Tuning...')
        start_time_tuning = time.time()
        site_best_model = hyperparameter_tuning_arima(site_series_train, 
                                                    max_p_P=(5,5), 
                                                    max_d_D=(3,3), 
                                                    max_q_Q=(10,10))
        tuning_exec_time = time.time() - start_time_tuning
        print("best model --> (p, d, q):", site_best_model.order, 
            " and  (P, D, Q, s):", site_best_model.seasonal_order)
        print("--- Tuning ended at %s seconds ---" % (tuning_exec_time))
        

        print('Training model...')
        start_time_training = time.time()
        site_model = sm.tsa.statespace.SARIMAX(
            site_series_train,
            order=site_best_model.order,
            seasonal_order=site_best_model.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        training_exec_time = time.time() - start_time_training
        print("--- Training ended at %s seconds ---" % (training_exec_time))
        
        
        print('Forecasting...')
        start_time_forecast = time.time()
        site_forecast = site_model.predict(start=len(site_series_train), 
                                        end=len(site_series_train)+len(site_series_test)-1,
                                        exog=None,
                                        dynamic=False)
        forecast_exec_time = time.time() - start_time_forecast
        print("--- Forecast ended at %s seconds ---\n" % (forecast_exec_time))
        
        save_groups_model_info(
            site_series_data, 
            site_best_model, 
            site_model,
            site_forecast,
            tuning=True,
            cross_validation=False
        )
        save_model_exec_times_info(
            tuning_exec_time, 
            training_exec_time,
            forecast_exec_time,
            tuning=True,
            cross_validation=False
        )
        
    overall_exec_time = time.time() - start_time_overall
    print("Multiple Training and Forecasting ended at--- %s seconds ---" % (overall_exec_time))
    write_model_info_serialized_file('{}{}_sarima_model_info.pkl'.format(PATH, DATASET_NAME))
    

def generate_gluon_forecasting_model(
    all_sites=False, 
    sites=None, 
    site_index=0
):
    print('\nGLUON MODEL\n')
    reset_model_info_dictionaries()
    
    if all_sites:
        print('Trainig and forecasting multiple time series groups...\n')
    else:
        print('Training and forecasting %s ...\n'%sites_names[site_index])
        
    start_time_overall = time.time()
    for site in tqdm(sites):
        print('\033[1m' + 'Group %s'%(site) + '\033[0m')
        site_df = pandas_dataframe_groupby_site.get_group(site)
        freq = 'D'
        site_train_test_timeseries_gluon = prepare_train_test_timeseries_gluon(site_df,
                                                                            29,
                                                                            data_column_name='OCCUPANCY_COUNT', 
                                                                            date_column_name='DATES', 
                                                                            freq=freq,
                                                                            start_day='2019-12-02')
        site_timeseries_gluon_train = site_train_test_timeseries_gluon[0]
        site_timeseries_gluon_test = site_train_test_timeseries_gluon[1]
        site_entry_train = next(iter(site_timeseries_gluon_train))
        site_entry_test = next(iter(site_timeseries_gluon_test))
        site_series_train = to_pandas(site_entry_train)
        site_series_test = to_pandas(site_entry_test)
        
        
        print('Transform dataset...')
        prediction_length = len(site_series_test) - len(site_series_train)
        transformation = create_transformation(freq, 2 * prediction_length, prediction_length)
        train_tf = transformation(iter(site_timeseries_gluon_train), is_train=True)
        train_tf_entry = next(iter(train_tf))
        print("--- Transforming has ended ---")
        

        print('Training model...')
        start_time_training = time.time()
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=prediction_length,
            context_length=2*prediction_length,
            freq=freq,
            trainer=Trainer(ctx="cpu",
                            epochs=5,
                            learning_rate=1e-3,
                            hybridize=False,
                            num_batches_per_epoch=50
                        )
        )
        site_predictor = estimator.train(site_timeseries_gluon_train)
        training_exec_time = time.time() - start_time_training
        print("--- Training ended at %s seconds ---" % (training_exec_time))
        
        
        print('Forecasting...')
        start_time_forecast = time.time()
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=site_timeseries_gluon_test,
            predictor=site_predictor,
            num_samples=100,
        )    
        site_forecasts = list(forecast_it)
        site_tss = list(ts_it)
        forecast_exec_time = time.time() - start_time_forecast
        print("--- Forecast ended at %s seconds ---\n" % (forecast_exec_time))    
        
        save_groups_model_info(
            (site_timeseries_gluon_train, site_timeseries_gluon_test),
            site_predictor, 
            (site_forecasts, site_tss),
            tuning=False,
            cross_validation=False
        )
        save_model_exec_times_info(
            training_exec_time, 
            forecast_exec_time,
            tuning=False,
            cross_validation=False
        )
        
    overall_exec_time = time.time() - start_time_overall
    print("Multiple Training and Forecasting ended at--- %s seconds ---" % (overall_exec_time))
    print('\nOverall_exec_time: {}'.format(overall_exec_time))
    print('groups_model_info: ', groups_model_info)
    print('exec_times_info: ', exec_times_info)
    write_model_info_serialized_file('{}{}_gluon_model_info.pkl'.format(PATH, DATASET_NAME))


if __name__ == "__main__":
    PATH = '../BDForecasting_V2/'
    DATA_PERIOD = 'D'
    DATASET_NAME = 'AON_1B'

    overall_exec_time = 0 
    groups_model_info = {
        'timeseries':[],
        'tuning_results':[],
        'models':[],
        'cross_validation_results':[],
        'forecasts':[]
    }
    exec_times_info = {
        'tuning_exec_times':[],
        'training_exec_times':[],
        'cross_validation_exec_times':[],
        'forecast_exec_times':[]
    }

    dataset_dataframes = dataset_preparation.generate_dataset_files(PATH, DATASET_NAME, DATA_PERIOD)
    df_grouped_dates = dataset_dataframes[0]
    series_grouped_dates = dataset_dataframes[1]
    pandas_dataframe_groupby_site = dataset_dataframes[2]
    sites_names = dataset_dataframes[3]

    all_sites = False
    site_index = 47
    sites = []

    if all_sites:
        sites = sites_names
    else:
        sites = [sites_names[site_index]]
        
    generate_fbprophet_forecasting_model(
        all_sites, sites, 
        site_index, 
        initial='5 days', 
        horizon='7 days'
    )
    generate_sarima_forecasting_model(all_sites, sites, site_index)
    generate_gluon_forecasting_model(all_sites, sites, site_index)