import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from funcs import *
from sklearn.ensemble import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

@st.cache
def load_data():
    california = fetch_california_housing()
    california_df = pd.DataFrame(data=np.c_[california['data'], california['target']], columns=california['feature_names'] + ['target'])
    return california_df, california


st.header('Вывод данных и графиков')

# Данные
data_load_state = st.text('Загрузка данных...')
data_df, data = load_data()
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(data_df.head())

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(data)

# Разделение выборки
test_size = st.sidebar.slider("test_size", 0.1, 0.9, value=0.3)
california_x_train, california_x_test, california_y_train, california_y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=1)

# Параметры моделей
n_estimators = st.sidebar.slider("n_estimators", 1, 15, value=5)
random_state = st.sidebar.slider("random_state", 1, 15, value=10)

#  Обучение моделей
br = BaggingRegressor(n_estimators=n_estimators, random_state=random_state)
br.fit(california_x_train, california_y_train)

adb = AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state)
adb.fit(california_x_train, california_y_train)

ext = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state)
ext.fit(california_x_train, california_y_train)

rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
rfr.fit(california_x_train, california_y_train)

gbr = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
gbr.fit(california_x_train, california_y_train)

# Отображение моделей
models = [br, adb, ext, rfr, gbr]                       # MODELS
mod = [i.__class__.__name__ for i in models]

modelss = st.sidebar.multiselect(
    "Choose algorithms", mod
)

# Метрики
metrics = [max_error, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error]
metr = [i.__name__ for i in metrics]

metricss = st.sidebar.multiselect("Choose metrics", metr)

# Данные для отображения models
figModels = []

for i in modelss:
    for j in models:
        if i == j.__class__.__name__:
            figModels.append(j)

# Данные для отображения metrics
figMetrics = []

for i in metricss:
    for j in metrics:
        if i == j.__name__:
            figMetrics.append(j)


fig = ENSEMBLES(figModels, figMetrics, california_x_test, california_y_test, 5, 0.4)
# plt.show()
st.pyplot(fig)