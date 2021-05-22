import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import sklearn as sk
from base import *
from funcs import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder
import tpot as tp
import autokeras as ak

def get_data_base(data, m_frac = 0.09):
    # Удаление пустых строк
    buff = data.dropna(axis=0)

    # Уникальные значения поля rating
    l = np.unique(buff[['rating']])
    # Выбираем все уникальные строки с определенным rating
    arrgroup = []
    for i in l:
        arrgroup.append(buff[buff['rating'] == i])

    # Новый фрейм
    newdata = pd.DataFrame()

    # Осуществляем выборку по каждому диапазону
    for i in arrgroup:
        newdata = newdata.append(i.sample(frac = m_frac))

    corr = newdata.corr()
    # Сумма пропущенных значений
    isnull = corr.isnull().sum()
    summNan = isnull.to_dict()

    keys = []
    index = 0
    # Если значение пропусков больше половины, то запоминае удаляемый столбец
    for key, val in summNan.items():
        if val >= buff.shape[1] / 2:
            keys.append(key)

    for i in keys:
        newdata = newdata.drop([i], axis = 1)

    return newdata

@st.cache
def load_data():
    return pd.read_csv('epi_r.csv', sep=",")

@st.cache
def load_data2():
    california = load_iris()
    california_df = pd.DataFrame(data=np.c_[california['data'], california['target']], columns=california['feature_names'] + ['target'])
    return california_df, california

menu = st.sidebar.selectbox(
    "Меню", ['Description', 'Main', 'AutoML']
)

if menu == 'Description':
    st.title("Описание базы данных")
    st.subheader("База данных 1")
    st.write("Данная база данных представляет собой мнформацию о рейтингах блюд, оцененных посетителями, основываях на ингредиентах, соднржащихся в них.")
    st.write("Задача: Классификация")
    
    st.subheader("База данных 2")
    st.write("Базовый датасет библиотеки sklearn (load_iris)")
    st.write("Задача: Классификация")

elif menu == 'Main':
    data = load_data()

    newdata = get_data_base(data)

    x = newdata.drop(['rating','title'], axis = 1)
    y = newdata[['rating']]
    mm = MinMaxScaler()
    x=pd.DataFrame(mm.fit_transform(x))
    y=pd.DataFrame(mm.fit_transform(y))       

    one = LabelEncoder()
    y = one.fit_transform(y.values.ravel())
    y = pd.DataFrame(y)

    st.subheader('Первые 5 значений')
    st.write(data.head())

    st.write('main')
    test_size = st.sidebar.slider("test_size", 0.1, 0.9, value=0.3)

    isModolsType = st.sidebar.selectbox(
        "Тип моделей", ['ensemble', 'svm']
    )

    if isModolsType == 'ensemble':
        n_estimators = st.sidebar.slider("n_estimators", 1, 15, value=5)

        # Models
        models = [
            AdaBoostClassifier, 
            BaggingClassifier, 
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier
        ]

        mod = [i.__name__ for i in models]

        modelss = st.sidebar.selectbox(
            "Choose ensemble algorithms", mod
        )

        # Metrics
        metrics = [
            accuracy_score,
            precision_score,
            recall_score,
            balanced_accuracy_score
        ]

        metr = [i.__name__ for i in metrics]

        metrs = st.sidebar.selectbox(
            "Choose metric", metr
        )

        CROSS = st.sidebar.checkbox('GridSearchCV ("n_estimators": np.arange(1, 10, 1))')
        CHECK = st.sidebar.button('Update')

        if CHECK:
            resmodel = ''
            for i in models:
                if i.__name__ == modelss:
                    resmodel = i

            resmetric = ''
            for i in metrics:
                if i.__name__ == metrs:
                    resmetric = i

            datalearn = [
                {
                    'model': resmodel(n_estimators=n_estimators, random_state=10),
                    'grid': {'n_estimators': np.arange(1, 10, 1)},
                    'metrics': [resmetric]
                }
            ]

            a = Analysing(datalearn)

            m1 = a.run(x, y, test_size=test_size)
            fig1 = m1.plot(size = [1, 5])
            st.pyplot(fig1)

            if CROSS:
                m2 = a.run(x, y, test_size=test_size, cross=True)
                fig2 = m2.plot_cross(size = [1, 5], space=0.8)
                st.pyplot(fig2)

    if isModolsType == 'svm':
        max_iter = st.sidebar.slider("max_iter", 1, 20, value=5)
        kernel = st.sidebar.selectbox(
            "Choose kernel algorithms", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        )

        # Models
        models = [
            LinearSVC, 
            SVC
        ]

        mod = [i.__name__ for i in models]

        modelss = st.sidebar.selectbox(
            "Choose svm algorithms", mod
        )

        # Metrics
        metrics = [
            accuracy_score,
            precision_score,
            recall_score,
            balanced_accuracy_score
        ]

        metr = [i.__name__ for i in metrics]

        metrs = st.sidebar.selectbox(
            "Choose metric", metr
        )

        CROSS = st.sidebar.checkbox('"max_iter": np.arange(1, 10, 1)')
        CHECK = st.sidebar.button('Update')

        if CHECK:
            resmodel = ''
            for i in models:
                if i.__name__ == modelss:
                    resmodel = i

            resmetric = ''
            for i in metrics:
                if i.__name__ == metrs:
                    resmetric = i
            
            if resmodel.__name__ == 'LinearSVC':
                resmodel = resmodel(max_iter=max_iter)
            else:
                resmodel = resmodel(kernel=kernel, max_iter=max_iter)

            datalearn = [
                {
                    'model': resmodel,
                    'grid': {'max_iter': np.arange(1, 10, 1)},
                    'metrics': [resmetric]
                }
            ]

            a = Analysing(datalearn)

            m1 = a.run(x, y, test_size=test_size)
            fig1 = m1.plot(size = [1, 5])
            st.pyplot(fig1)

            if CROSS:
                m2 = a.run(x, y, test_size=test_size, cross=True)
                fig2 = m2.plot_cross(size = [1, 5], space=0.8)
                st.pyplot(fig2)

elif menu == 'AutoML':
    st.write('AutoML')
    data_load_state = st.text('Загрузка данных...')
    data_df, data = load_data2()
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
    br = BaggingClassifier(n_estimators=n_estimators, random_state=random_state)
    br.fit(california_x_train, california_y_train)

    adb = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    adb.fit(california_x_train, california_y_train)

    ext = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
    ext.fit(california_x_train, california_y_train)

    rfr = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rfr.fit(california_x_train, california_y_train)

    gbr = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    gbr.fit(california_x_train, california_y_train)

    # Отображение моделей
    models = [br, adb, ext, rfr, gbr]                       # MODELS
    mod = [i.__class__.__name__ for i in models]

    modelss = st.sidebar.multiselect(
        "Choose algorithms", mod
    )

    # Метрики
    metrics = [accuracy_score, balanced_accuracy_score]
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

    CHECK = st.sidebar.checkbox("AutoML")
    
    if CHECK:
        tpp = tp.TPOTClassifier(generations=2, verbosity=2, max_time_mins=10)
        res = tpp.fit(california_x_train, california_y_train)
        st.write(res.best)