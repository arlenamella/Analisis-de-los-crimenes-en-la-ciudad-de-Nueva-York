import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV



# Gráfico distribuciones de variables

"""
Grafica la distribución de las variables dentro de un dataframe, según su tipo de dato.
@params
   df: DataFrame a graficar
   columns= cantidad de columnas en las que se visualizaran los graficos en el jupyter, 
    predeterminado en 3 columnas.
"""

def distribution_plots(df, columns=3):
    
    rows = np.ceil(df.shape[1] / columns).astype(int)
    height = rows * 3.5
    fig = plt.figure(figsize=(12, height))
 
    for n, i in enumerate(df.columns):
        
        if df[i].dtype in ('object', 'int64') :
            fig.add_subplot(rows, columns, n+1)
            ax = sns.countplot(x=i, data=df)
            plt.title(i)
            plt.xlabel('')
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + .5,
                    '{:1.2f}'.format(height/len(df[i])), ha="center")

        if df[i].dtype == 'float64':
            fig.add_subplot(rows, columns, n+1)
            ax = sns.distplot(df[i])
            plt.title(i)
            plt.xlabel('')
            
    plt.tight_layout()

    return

# Función de preprocesamiento de base

def pre_processing(df, num_cols, obj_cols, exclude, target, std_scaler=False, 
                    one_hot=True, custom_split=True):
    """
    Función para aplicar un preprocesamiento de datos sobre un dataframe

    """

    #columns = [target]+num_cols+obj_cols+exclude
    columns = num_cols+obj_cols+exclude
    tmp = df[columns]
    
    num_steps, obj_steps = None, None
    
    if std_scaler:
        num_steps = StandardScaler()
    if one_hot:
        # Categorías para one-hot
        tmp2 = tmp[obj_cols]
        categories = [list(tmp2[var].value_counts().sort_values(ascending=False).index) for var in tmp2]
        obj_steps = OneHotEncoder(categories, sparse=False, drop='first')
        #Nombre de columnas dummy
        tuples = [(var, list(tmp2[var].value_counts().sort_values(ascending=False).index)[1:]) for var in tmp2]
        dummy_names = ['{}_{}'.format(tup[0], cat) for tup in tuples for cat in tup[1]]
        # Renombre de columnas df final
        #columns = [target]+num_cols+dummy_names+exclude
        columns = num_cols+dummy_names+exclude
        target = ['{}_{}'.format(tup[0], tup[1][0]) for tup in tuples if tup[0] == target][0]

    # Creación de pipeline para preproceso
    num_pipe = make_pipeline(num_steps)
    obj_pipe = make_pipeline(obj_steps)

    column_transformer = make_column_transformer(
                        (OrdinalEncoder(), target),
                            (num_pipe, num_cols),
                            (obj_pipe, obj_cols),
                            ('passthrough', exclude))

    preprocessed = column_transformer.fit_transform(tmp)

    df_pre = pd.DataFrame(data=preprocessed, columns=columns)
    
     # Train test split
    if custom_split:
   
        df_train = df_pre[df_pre['sample'] == 'train']
        df_test = df_pre[df_pre['sample'] == 'test']

        X_train = df_train.drop(columns=[target, 'sample'])
        y_train = df_train[target]

        X_test = df_test.drop(columns=[target, 'sample'])
        y_test = df_test[target]
    
    else:
        X = df_pre.drop(columns=[target, 'sample'])
        y = df.pre[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)

    # Eliminación de columnas con una clase
    to_delete = [var for var in X_train if len(X_train[var].unique()) == 1]
    X_train = X_train.drop(columns=to_delete)
    X_test = X_test.drop(columns=to_delete)

    # Aegurar tipo de dato para entrenar y modelos modelos
    X_train = X_train.astype('int')
    y_train = y_train.astype('int')
    X_test = X_test.astype('int')
    y_test = y_test.astype('int')

    return X_train, X_test, y_train, y_test


# Métricas de problema de clasificación

def clf_metrics(clf, X_train, y_train, X_test, y_test):
    """
    Imprime un reporte con las métricas de problemas de clasificación clásicas:
    @params
        clf: modelo a graficar sus métricas.
        X_train: Set de datos de entrenamiento.
        y_train: Vector objetivo de  entrenamiento.
        X_test: Set de datos de validación.
        y_test: Vector objetivo de validación.
    """    
    tic = time()
    # Corrroboración preproceso target
    if (y_train.dtype =='object') & (y_test.dtype == 'object'):

        lbl_encoder = LabelEncoder()
        y_train = lbl_encoder.fit_transform(y_train)
        y_test = lbl_encoder.transform(y_test)

    # Entrenar el modelo
    clf.fit(X_train, y_train)
    # Imprimir mejores parámetros sí el objeto 
    if isinstance(clf, GridSearchCV):
        print(clf.best_params_)
    # Predecir la muestra de validación
    y_hat = clf.predict(X_test)
    # Métricas
    metrics = {'ROC_Score': roc_auc_score(y_test, y_hat).round(3),
               'Confusion_Matrix': confusion_matrix(y_test, y_hat).round(3),
               'Classification_Report': classification_report(y_test, y_hat)}
    for key, value in metrics.items():
        print('{}:\n{}'.format(key, value))
    return print("Realizado en {:.3f}s".format(time() - tic))


def compare_classifiers(estimators, X_test, y_test, n_cols=2):

    """
    Compara en forma gráfica las métricas de clasificación a partir de una lista de 
    tuplas con los modelos (nombre_modelo, modelo_entrendo)
    @params
        estimators: lista de tuplas con los modelos.
        X_test: Set de datos de validación.
        y_test: Vector objetivo de validación.
        n_cols= número de columnas en las que se visualizaran los 
            gráficos en el jupyter, predeterminado en 2.
    """

    rows = np.ceil(len(estimators)/n_cols)
    height = 2 * rows
    width = n_cols * 5
    fig = plt.figure(figsize=(width, height))

    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for n, model in enumerate(estimators):

        y_hat = model[1].predict(X_test) 
        dc = classification_report(y_test, y_hat, output_dict=True)

        plt.subplot(rows, n_cols, n + 1)

        for i, j in enumerate(['0', '1', 'macro avg']):

            tmp = {'0': {'marker': 'x', 'label': f'Class: {j}'},
                   '1': {'marker': 'x', 'label': f'Class: {j}'},
                   'macro avg': {'marker': 'o', 'label': 'Avg'}}

            plt.plot(dc[j]['precision'], [1], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['recall'], [2], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['f1-score'], [3], marker=tmp[j]['marker'],color=colors[i], label=tmp[j]['label'])
            plt.axvline(x=.5, ls='--')

        plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])
        plt.title(model[0])
        plt.xlim((0.1, 1.0))

        if (n + 1) % 2 == 0:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    fig.tight_layout()
    
    return

def distribution_plots(df, columns=3):
    
    rows = np.ceil(df.shape[1] / columns).astype(int)
    height = rows * 3.5
    fig = plt.figure(figsize=(12, height))
 
    for n, i in enumerate(df.columns):
        
        if df[i].dtype in ('object', 'int64') :
            fig.add_subplot(rows, columns, n+1)
            ax = sns.countplot(x=i, data=df)
            plt.title(i)
            plt.xlabel('')
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + .5,
                    '{:1.2f}'.format(height/len(df[i])), ha="center")

        if df[i].dtype == 'float64':
            fig.add_subplot(rows, columns, n+1)
            ax = sns.distplot(df[i])
            plt.title(i)
            plt.xlabel('')
            
    plt.tight_layout()

    return

def distribution_plots_cat(df, columns=1):
    
    rows = np.ceil(df.shape[1] / columns).astype(int)
    height = rows * 3.5
    fig = plt.figure(figsize=(12, height))
 
    for n, i in enumerate(df.columns):
        
        if df[i].dtype in ('object', 'int64') :
            fig.add_subplot(rows, columns, n+1)
            ax = sns.countplot(x=i, data=df)
            plt.title(i)
            plt.xlabel('')
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + .5,
                    '{:1.2f}'.format(height/len(df[i])), ha="center")

    plt.tight_layout()

    return