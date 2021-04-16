import pandas as pd
import numpy as np
from operator import itemgetter

def splitter(x, y, train_part = 80):
    df = x.join(y)

    test_df = df.sample(frac = (100-train_part)/100)

    massindexs = []
    for i, val in test_df.iterrows():
        massindexs.append(i)
    df = df.drop(df.index[massindexs])

    # return df, test_df
    return df[[df.columns[0]]], df[[df.columns[1]]], test_df[[test_df.columns[0]]], test_df[[test_df.columns[1]]]

class KNN():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def sort_for_key(self, el):
        return val

    def predict(self, test_x, k = 2):
        prediction_y = []
        for i, val in test_x.iterrows():
            elementsmin = []
            elementsmax = []
            df1 = self.x[self.x[self.x.columns[0]] < val[0]]
            df2 = self.x[self.x[self.x.columns[0]] > val[0]]
            df1 = df1[::-1]

            j = 0
            for i1, val1 in df1.iterrows():
                if j < k:
                    elementsmin.append([i1, val1[0]])
                    j+=1

            j = 0
            for i2, val2 in df2.iterrows():
                if j < k:
                    elementsmax.append([i2, val2[0]])
                    j+=1

            elementsmin = list(reversed(elementsmin))
            new_arr = elementsmin + elementsmax
            new_res = ''

            if len(new_arr) > k:
                num = k - len(new_arr)
                if num % 2 == 0:
                    left = int(k / 2)
                    new_res = new_arr[left:len(new_arr)-left]
                else:
                    left = int(k / 2)
                    new_res = new_arr[left:len(new_arr)]
            else:
                new_res = new_arr

            ys = []
            for m in new_res:
                data = self.y[self.y.index == m[0]]
                for p, ydata in data.iterrows():
                    ys.append(ydata[0])

            ys = np.unique(ys)
            mass = {}

            for b in ys:
                for o in new_res:
                    res = self.y[self.y.index == o[0]]
                    if res[res.columns[0]].values[0] == b:
                        if b not in mass:
                            mass.update({b:0})
                        mass[b] += o[1]
            
            mass = sorted(mass.items(), key=lambda item: item[1])
            if len(mass) != 0:
                prediction_y.append(mass[len(mass)-1][0])
                
        return np.array(prediction_y)