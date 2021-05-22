from enum import Enum
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import *
import seaborn as sns
import matplotlib. pyplot as plt
sns.set(style="ticks")
from sklearn.datasets import *
import sklearn

class Analysing():
    # Меняемые переменные
    cv = 5
    test_size = 0.3

    task = 0
    models = 0
    scores = {}

    def __init__(self, models):
        self.models = models

    def run(self, x, y, test_size = 0.5, cross = False, cv = 5):
        self.test_size = test_size
        self.cv = cv
        self.scores = {}
        if not cross:
            self.fit_no_cross(x, y)
        else:
            self.fit_cross(x, y, cv)
        return self
        

    def fit_no_cross(self, x, y):
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=self.test_size, random_state=1)
            for i in self.models:
                m = i['model']
                m.fit(train_x, train_y)
                result = m.predict(test_x)
                self.scores.update({m: {}})
                for j in i['metrics']:
                    if j.__name__ == 'precision_score' or j.__name__ == 'recall_score':
                        self.scores[m].update({j : j(result, test_y, average="micro")})
                    else:
                        self.scores[m].update({j : j(result, test_y)})

    def fit_cross(self, x, y, cv):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=self.test_size, random_state=1)
        for i in self.models:
            for j in i['metrics']:
                make_s = make_scorer(j)
                grid = GridSearchCV(i['model'], i['grid'], cv = self.cv, scoring = make_s)
                grid.fit(x, y.values.ravel())
                # m.fit(train_x, train_y.values.ravel())
                # result = m.predict(test_x)

                if j not in self.scores:
                    self.scores.update({j: {}})
                
                self.scores[j].update({str(grid.best_estimator_) : grid.best_score_})

    def plot(self, size = [4, 7], space = 0.6):
        rows= len(self.scores)
        columns = 1
        index = 1
        fig = plt.figure(figsize=(size[1], (size[0] + space)*rows))
        fig.subplots_adjust(hspace=space)
        
        for key, val in self.scores.items():
            x = []
            y = []
            for j, jval in val.items():
                m_list = j.__name__.split('_')
                m_str = ''
                for k in m_list:
                    m_str += k + '\n'
                x.append(m_str)
                y.append(jval)
            pos = np.arange(len(val.items()))
            ax = fig.add_subplot(rows, columns, index)
            ax.barh(np.array(x), np.array(y), align='center')
            ax.set_title(key)
            for a,b in zip(pos, y):
                ax.text(0.1, a-0.1, str(round(b,3)), color='pink')
            index+=1
        return fig

    def plot_cross(self, size = [7, 4], space = 0.8):
        columns=1
        rows=len(self.scores)
        index = 1
        fig = plt.figure(figsize=(size[1], (size[0] + space)*columns))
        fig.subplots_adjust(hspace=space)
        
        for key, val in self.scores.items():
            x = []
            y = []
            for j, jval in val.items():
                m_str = ''
                separr = j.split('(')
                for k in separr:
                    m_str += k +'\n'

                x.append(m_str)
                y.append(jval)
            pos = np.arange(len(val.items()))
            ax = fig.add_subplot(rows, columns, index)
            ax.barh(np.array(x), np.array(y), align='center')

            ax.set_title(key.__name__)

            for a,b in zip(pos, y):
                ax.text(0.1, a-0.1, str(round(b,3)), color='pink')
            index+=1

'''
[
    {
        'model': name,
        'grid': gridsearchCV : {},
        'metrics': m : []
    }
]
'''

if __name__ == "__main__":
    data = load_iris()
    data_df = pd.DataFrame(data=np.c_[data['data']], columns=data['feature_names'])
    data_df['target'] = data.target

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)

    metrics = [accuracy_score, balanced_accuracy_score] # mean_absolute_error, mean_squared_errormedian_absolute_error
    models = [
        {
            'model': KNeighborsClassifier(n_neighbors = 7),
            'grid': {'n_neighbors': np.arange(1,10,1)},
            'metrics': metrics
        }
    ]
    a = Analysing(models)

    x = pd.DataFrame(data['data'])
    y = pd.DataFrame(data['target'])
    
    # for i in sklearn.metrics.SCORERS.keys():
    #     print(i)
    m1 = a.run(x, y)
    m1.plot()
    plt.show()
    m2 = a.run(x, y, cross=True)
    m2 = a.plot_cross()
    plt.show()
