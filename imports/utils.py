EPSILON = 1e-10

'''
Plot train and test spliting
:parameter
    :param ts_train: pandas Series
    :param ts_test: pandas Series
'''
def plot_train_test(ts_train, ts_test, figsize=(15,5), fontsize=10, rotation=45):
    fig, ax = plt.subplots(nrows=1, 
                           ncols=2, 
                           sharex=False, 
                           sharey=True, 
                           figsize=figsize)
    
    ts_train.plot(ax=ax[0], 
                  grid=True,
                  title="Train", 
                  color="orange",
                  fontsize=fontsize)
    ax[0].tick_params(axis='x', rotation=rotation)
    
    ts_test.plot(ax=ax[1], 
                 grid=True, 
                 title="Test",
                 color="blue",
                 fontsize=fontsize)
    ax[1].tick_params(axis='x', rotation=rotation)
    
    ax[0].set(xlabel=None)
    ax[1].set(xlabel=None)
    plt.show()


'''
Split train/test from any given data point.
:parameter
    :param ts: pandas Series
    :param test: num or str - test size (ex. 0.20) or index position
                 (ex. "yyyy-mm-dd", 1000)
:return
    ts_train, ts_test
'''
def split_train_test(ts, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[ 
                      ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
    print("--- splitting at index: ", split, "|", 
          ts.index[split], "| test size:", perc, " ---")
    
    ## split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)
        
    return ts_train, ts_test


'''
Evaluation metrics for predictions.
:parameter
    :param dtf: DataFrame with columns raw values, fitted training  
                 values, predicted test values
:return
    dataframe with raw ts and forecast
'''
def utils_evaluate_forecast(dtf, title, plot=True, figsize=(20,13)):
    try:
        ## residuals
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]
        
        ## kpi
        residuals_mean = dtf["residuals"].mean()
        residuals_std = dtf["residuals"].std()
        error_mean = dtf["error"].mean()
        error_std = dtf["error"].std()
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  
        mse = dtf["error"].apply(lambda x: x**2).mean()
        rmse = np.sqrt(mse)  #root mean squared error
        
        ## intervals
        dtf["conf_int_low"] = dtf["forecast"] - 1.96*residuals_std
        dtf["conf_int_up"] = dtf["forecast"] + 1.96*residuals_std
        dtf["pred_int_low"] = dtf["forecast"] - 1.96*error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96*error_std
        
        ## plot
        if plot==True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)      
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax2.fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)     
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))
        
        return dtf[["ts","model","residuals","conf_int_low","conf_int_up", 
                    "forecast","error","pred_int_low","pred_int_up"]]
    
    except Exception as e:
        print("--- got error ---")
        print(e)


'''
Fit SARIMAX (Seasonal ARIMA with External Regressors):  
    y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] + 
              b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d: 
                  degree of differencing (to remove trend), q: order 
                  of moving average (MA)
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of 
                  observations per seasonal (ex. 7 for weekly 
                  seasonality with daily data, 12 for yearly 
                  seasonality with monthly data)
    :param exog_train: pandas dataframe or numpy array
    :param exog_test: pandas dataframe or numpy array
:return
    dtf with predictons and the model
'''
def fit_sarimax(ts_train, ts_test, order=(1,0,1), 
                seasonal_order=(0,0,0,0), exog_train=None, 
                exog_test=None, figsize=(15,10)):
    ## train
    model = smt.SARIMAX(ts_train, order=order, 
                        seasonal_order=seasonal_order, 
                        exog=exog_train, enforce_stationarity=False, 
                        enforce_invertibility=False).fit()
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), 
                            end=len(ts_train)+len(ts_test)-1, 
                            exog=exog_test)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "ARIMA "+str(order) if exog_train is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_arctangent_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true+EPSILON))))


def get_performance_metrics_training(cross_validation_results):
    performance_overall_metrics = performance_metrics(cross_validation_results)
    mse_mean = np.mean(performance_overall_metrics['mse'])
    rmse_mean = np.mean(performance_overall_metrics['rmse'])
    mae_mean = np.mean(performance_overall_metrics['mae'])
    
    return mse_mean, rmse_mean, mae_mean


def get_performance_metrics_forecast(y_true, y_pred):
    mse_mean = mean_squared_error(y_true, y_pred)
    rmse_mean = sqrt(mse_mean)
    mae_mean = mean_absolute_error(y_true, y_pred)
    maape = mean_arctangent_absolute_percentage_error(y_true, y_pred)
    
    return mse_mean, rmse_mean, mae_mean, 


def replace_negative_values_with_ceros(df, forecast_df=False):
    df['yhat'] = df['yhat'].apply(lambda x: 0 if x<0 else x)
    
    if forecast_df:
        return df['yhat']
    
    return df


def scale_data(df, scaled_column_name='y'):
    scaler = MinMaxScaler()
    occ_values_to_scale = df[scaled_column_name].values.reshape(-1,1)
    scaler_fit = scaler.fit(occ_values_to_scale)
    df_copy = copy.deepcopy(df)
    df_copy.loc[:,scaled_column_name] = (scaler_fit.transform(occ_values_to_scale))
    
    return df_copy, scaler
    
    
def prepare_timeserie_df(df, pd_series=False, series_index=None, perform_scale_data=True):
    transformed_data = None
    scaler = None
    site_scaled_data = None
    
    if pd_series:
        if perform_scale_data:
            scaled_data = scale_data(df, 'OCCUPANCY_COUNT')
            site_sacaled_data = scaled_data[0]
            scaler = scaled_data[1]
            transformed_data = site_sacaled_data['OCCUPANCY_COUNT']
        else:
            transformed_data = df['OCCUPANCY_COUNT']
        transformed_data.index = series_index
    
    else:
        df_grouped_dates = df.groupby(['DATES'], as_index=False).sum()
        transformed_data = pd.DataFrame({'ds':df_grouped_dates['DATES'], 'y':df_grouped_dates['OCCUPANCY_COUNT']})
        if perform_scale_data:
            scaled_data = scale_data(transformed_data)
            transformed_data = scaled_data[0]
            scaler = scaled_data[1]
            
    return transformed_data, scaler


def prepare_ListDataset_gluon(target, start, feat_static_cat, freq):
    timeserie_gluon = ListDataset([{FieldName.TARGET: target,
                                    FieldName.START: start,
                                    FieldName.FEAT_STATIC_CAT: [feat_static_cat]}
                                  ], freq=freq)
    
    return timeserie_gluon


def prepare_train_test_timeseries_gluon(df,
                                        cut_index,
                                        data_column_name='OCCUPANCY_COUNT', 
                                        date_column_name='DATES', 
                                        freq='H',
                                        perform_scale_data=True,
                                        start_day=None):
    scaler = None
    
    if perform_scale_data:
        scaled_data = scale_data(df, scaled_column_name=data_column_name)
        df = scaled_data[0]
        scaler = scaled_data[1]
    
    target = np.array(df[data_column_name])
    start = pd.Timestamp(start_day, freq=freq)
    feat_static_cat = np.array(0)
    
    timeserie_gluon_train = prepare_ListDataset_gluon(target[:cut_index], start, feat_static_cat, freq)
    timeserie_gluon_test = prepare_ListDataset_gluon(target, start, feat_static_cat, freq)
    
    return timeserie_gluon_train, timeserie_gluon_test, scaler


def hyperparameter_tuning(df, 
                          site_name, 
                          cutoffs=[], 
                          seasonality_prior_scale=[], 
                          changepoint_prior_scale=[],
                          horizon=None):
    if not cutoffs:
        cutoffs = pd.to_datetime(['2018-12-31', '2019-05-30', '2019-09-30'])
    if not changepoint_prior_scale:        
        changepoint_prior_scale = [0.001, 0.01, 0.1, 0.5]
    if not seasonality_prior_scale:
        seasonality_prior_scale = [0.01, 0.1, 1.0, 10.0]
        
    param_grid = {
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale}

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []

    for params in tqdm(all_params, desc="Tuning: %s"%site_name):
        p = Prophet(**params
                   ).add_seasonality(name='monthly', 
                                     period=30.5, 
                                     fourier_order=5, 
                                     prior_scale=0.02
                                     ).fit(df)
        df_cv = cross_validation(p, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    
    best_parameters = tuning_results[tuning_results['rmse'] == tuning_results['rmse'].min()]
    seasonality_prior_scale = best_parameters['seasonality_prior_scale']
    changepoint_prior_scale = best_parameters['changepoint_prior_scale']
    
    return seasonality_prior_scale, changepoint_prior_scale


def hyperparameter_tuning_arima(series_train, 
                                max_p_P=(10,10), 
                                max_d_D=(3,3), 
                                max_q_Q=(10,10),
                                max_order=20,
                                m=7,
                                info_criterion='aic'):
    best_model = pmdarima.auto_arima(series_train, exogenous=None,                      
                                    seasonal=True, stationary=True, 
                                    m=m, information_criterion=info_criterion, 
                                    max_order=max_order,                                     
                                    max_p=max_p_P[0], max_d=max_d_D[0], max_q=max_q_Q[0],                       
                                    max_P=max_p_P[1], max_D=max_d_D[1], max_Q=max_q_Q[1],                       
                                    error_action='ignore')

    return best_model

def parser_date(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def create_transformation(freq, context_length, prediction_length):
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=2),
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[
                    FieldName.FEAT_AGE,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
        ]
    )


def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()