import pandas as pd
import pickle
import tqdm
import matplotlib.pyplot as plt
import warnings
from itertools import combinations

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

warnings.simplefilter(action='ignore')

def convert_pickle_to_df(name, dest_name = None, save = False):
    with open(name, 'rb') as f:
        experiments_results = pickle.load(f)
    
    parameters = list(experiments_results[1]['parameters'].keys())
    metrics = list(experiments_results[1]['val_results'].columns)
    metrics.remove('index')
    metrics_names = [f'{metric} best epoch' for metric in metrics]

    experiments_df = pd.DataFrame(columns = parameters+metrics+metrics_names)

    for i in tqdm.tqdm(experiments_results.keys()):
        temp = dict()
        for par in parameters:
            par_value = experiments_results[i]['parameters'][par]
            par_value = str(par_value) if isinstance(par_value, list) else par_value
            temp[par] = par_value
        for metric, best_epoch in zip(metrics, metrics_names):
            if isinstance(experiments_results[i]['val_results'], list):
                experiments_results[i]['val_results'] = experiments_results[i]['val_results'][0]
            temp[metric] = experiments_results[i]['val_results'][metric].max()
            temp[best_epoch] = experiments_results[i]['val_results'][metric].argmax()+1
        temp = pd.DataFrame(temp, index=[i])
        experiments_df = pd.concat([experiments_df.loc[:], temp])

    if save:
        if dest_name is None:
            dest_name = f'{name.split(".pickle")[0]}.csv'
        experiments_df.to_csv(dest_name)
            
    return experiments_df

def filter_df(df, exclude):
    for j in exclude.keys():
        df = df[df[j].isin(exclude[j])]
    return df

def plot_metric(df, metric = 'macro avg'):
    df[metric].hist()
    plt.title(metric)
    plt.show()

def plot_all_metrics(df):
    for i in [0, 1, 'macro avg', 'weighted avg', 'accuracy']:
        plot_metric(df, metric = i)

def plot_metric_by_parameter(df, parameter, metric='macro avg', lim=False):
    df.boxplot(column=metric, by=parameter, figsize=(12,12))
    if lim:
        plt.ylim([0.7, df[metric].max()+0.01])
    plt.title(metric)
    plt.show()

def plot_metric_by_all_parameters(df, metric='macro avg', parameters=None):
    metrics = ['0', '1', 'macro avg', 'weighted avg', 'accuracy']
    metrics = metrics + [f'{m} best epoch' for m in metrics]
    if parameters is None:
        parameters = [i for i in df.columns if i not in metrics]
    parameters = powerset(parameters)
    for i in parameters:
        if len(i)>0:
            plot_metric_by_parameter(df, list(i), metric=metric)

def plot_all_metrics_by_all_parameters(df):
    metrics = [0, 1, 'macro avg', 'weighted avg', 'accuracy']
    epochs = [f'{m}_best_epoch' for m in metrics]
    parameters = [i for i in parameters if i not in metrics and i not in epochs]
    for m in metrics:
        plot_metric_by_all_parameters(df, metric=m)

def plot_metric_by_parameter_difference(df, parameter, metric='macro avg', parameter_values=None):
    if parameter_values is None:
        parameter_values = list(combinations(list(df[parameter].unique()), 2))

    for i,j in parameter_values:
        res = df[metric][df[parameter]==i].values - df[metric][df[parameter]==j].values
        pd.Series(res).hist()
        plt.title(f'{parameter} {i} and {j} difference')
        plt.show()
