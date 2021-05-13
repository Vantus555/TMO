from io import StringIO 
from IPython.display import Image
import graphviz 
import pydotplus
from sklearn.tree import export_graphviz

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib. pyplot as plt
sns.set(style="ticks")

# Визуализация дерева
def get_png_tree(tree_model_param, feature_names_param):
    dot_data = StringIO()
    export_graphviz(tree_model_param, out_file=dot_data, feature_names=feature_names_param,
                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph.create_png()

def view_tree(tree_model_param, feature_names_param):
    Image(get_png_tree(tree_model_param, feature_names_param))

def vis_models_quality(array_metric, array_labels, str_header, figsize=(5, 5)):
    fig, ax1 = plt.subplots(figsize=figsize)
    pos = np.arange(len(array_metric))
    rects = ax1.barh(pos, array_metric,
                     align='center',
                     height=0.5, 
                     tick_label=array_labels)
    ax1.set_title(str_header)
    for a,b in zip(pos, array_metric):
        plt.text(0.2, a-0.1, str(round(b,3)), color='white')

def ENSEMBLES(models : [], metrics : [], x_test, y_test, size = 5, space = 0.2):
    columns=len(models)
    rows=1
    index = 1
    fig = plt.figure(figsize=((size + space)*columns, size))
    fig.subplots_adjust(wspace=space)
    pos = np.arange(len(models))

    for name in metrics:
        x = []
        y = []
        for func in models:
            data = func.predict(x_test)
            x.append(func.__class__.__name__)
            y.append(name(data, y_test))


        ax = fig.add_subplot(rows, columns, index)
        ax.barh(np.array(x), np.array(y), align='center')
        ax.set_title(name.__name__)
        for a,b in zip(pos, y):
            ax.text(0.1, a-0.1, str(round(b,3)), color='white')
        index+=1

    # for func in models:
    #     x = []
    #     y = []
    #     data = func.predict(x_test)
    #     for name in metrics:
    #         x.append(name.__name__)
    #         y.append(name(data, y_test))
    #     ax = fig.add_subplot(rows, columns, index)
    #     ax.barh(np.array(x), np.array(y), align='center')
    #     ax.set_title(func.__class__.__name__)
    #     for a,b in zip(pos, y):
    #         ax.text(0.1, a-0.1, str(round(b,3)), color='white')
    #     index+=1
        
    