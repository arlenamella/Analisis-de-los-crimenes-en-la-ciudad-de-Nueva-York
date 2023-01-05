
import numpy as np
import pandas as pd
import pickle
import re
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from sklearn.metrics import classification_report, f1_score, accuracy_score

parse_model_name = lambda x: re.sub("'>'", "",
                                    str(x.estimator.__class__).split('.')[-1])

def plot_model_performance(model, param_n=20, error_metric=True):
    # preserve best param combination
    best_params = str(model.best_params_)
    # convert params combination to string and preserve
    # gather train and test scores, and params names, convert to dataframe
    tmp_df = pd.DataFrame({'params':list(map(lambda x: str(x), model.cv_results_['params'])),
                           'train_score':model.cv_results_['mean_train_score'],
                           'test_score': model.cv_results_['mean_test_score']})
    # sort
    tmp_df = tmp_df.sort_values(by='test_score', ascending=False)
    
    if error_metric is True:
        penalizer = 1
    else:
        penalizer = 0
    
    # limit size 
    if tmp_df.shape[0] > param_n:
        tmp_df = tmp_df.head(param_n)
    
    tmp_df['signal_optim'] = np.where(tmp_df['params'] == best_params, 'tomato', 'slategrey')
    
    tmp_grid = GridSpec(2, 1, height_ratios=[1, 4])
    
    plt.subplot(tmp_grid[0])
    plt.hist(penalizer - tmp_df['train_score'],
             label='Training Scores',
             alpha=.5)
    plt.hist(penalizer - tmp_df['test_score'],
             label = 'Testing Scores',
             alpha=.5)
    plt.legend()
    
    sns.despine()
    plt.subplot(tmp_grid[1])
    plt.scatter(penalizer - tmp_df['test_score'], tmp_df['params'], color=tmp_df['signal_optim'], label='Testing Scores', marker='*')
    plt.scatter(penalizer - tmp_df['train_score'], tmp_df['params'], color=tmp_df['signal_optim'], label='Training Scores', marker='o')
    plt.grid(axis='y', linestyle='--', lw=.2)
    plt.legend()
    sns.despine()
    plt.suptitle('{}\nBest hyperparams: {}\n Test Error:{}'.format(parse_model_name(model),model.best_params_, (1 - model.best_score_.round(2))))
    
    
def plot_benchmark(models):
    # placeholder
    tmp_xaxis = []
    
    for index, values in enumerate(models):
        # preserve scikit-learn model name
        tmp_xaxis.append(re.sub("'>","",str(values.estimator.__class__).split('.')[-1]))
        # Visualize each test iteration
        plt.plot(values.cv_results_['mean_test_score'],
                 [index + .90] * len(values.cv_results_['params']),
                 'o', color='lightblue', alpha=.2)
        # Visualize each train iteration
        plt.plot(values.cv_results_['mean_train_score'],
                 [index + 1.10] * len(values.cv_results_['params']),
                 'o', color='pink', alpha=.2)
        # Signal test mean values
        plt.scatter(np.mean(values.cv_results_['mean_test_score']),
                    [index + .90], marker='|', s=100,
                    color='dodgerblue',zorder=3, label='Test')
        # Signal train mean values
        plt.scatter(np.mean(values.cv_results_['mean_train_score']),
                    [index + 1.10], marker='|', s=100,
                    color='tomato',zorder=3, label='Train')
        # Signal best performance on test
        plt.scatter(values.best_score_,
                    [index + 1], color='purple',
                    zorder=10)
        
    plt.yticks(range(1,len(tmp_xaxis) + 1), tmp_xaxis)
    plt.grid(axis='y')
    plt.legend(handles=[Line2D([0], [0], color='dodgerblue', marker="|", label='Test'),
                        Line2D([0], [0], color='tomato', marker='|', label='Training')])
        
        
def plot_classification_report(y_true, y_hat):
    """TODO: Docstring for plot_classification_report.
    :y_true: TODO
    :y_hat: TODO
    :returns: TODO
    """
    # process string and store in a list
    report = classification_report(y_true, y_hat).split()
    # keep values
    report = [i for i in report if i not in ['precision', 'recall', 'f1-score', 'support', 'avg']]
    # transfer to a DataFrame
    report = pd.DataFrame(np.array(report).reshape(len(report) // 5, 5))
    # asign columns labels
    report.columns = ['idx', 'prec', 'rec', 'f1', 'n']
    # preserve class labels
    class_labels = report.iloc[:np.unique(y_true).shape[0]].pop('idx').apply(int)
    # separate values
    class_report = report.iloc[:np.unique(y_true).shape[0], 1:4]
    # convert from str to float
    class_report = class_report.applymap(float)
    # convert to float average report
    average_report = report.iloc[-1, 1: 4].apply(float)

    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for i in class_labels:
        plt.plot(class_report['prec'][i], [1], marker='x', color=colors[i])
        plt.plot(class_report['rec'][i], [2], marker='x', color=colors[i])
        plt.plot(class_report['f1'][i], [3], marker='x',color=colors[i], label=f'Class: {i}')

    plt.scatter(average_report, [1, 2, 3], marker='o', color='forestgreen', label='Avg')
    plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])