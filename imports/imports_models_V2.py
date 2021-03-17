import copy
from datetime import datetime
import json
import itertools
from itertools import islice
import math
from math import sqrt
from pathlib import Path
import pickle
import os
import time
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from multiprocessing import Pool, cpu_count
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf
import pmdarima
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.mx.trainer import Trainer
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot, plot_components_plotly, plot_cross_validation_metric, plot_plotly
from fbprophet.diagnostics import cross_validation, performance_metrics
from plotly import offline as py
from plotly import graph_objs as go
import mxnet as mx
from mxnet import gluon
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import AddAgeFeature,AddObservedValuesIndicator,Chain
from gluonts.transform import ExpectedNumInstanceSampler,InstanceSplitter,SetFieldIfNotPresent
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator