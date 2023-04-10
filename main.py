import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

tabela = pd.read_csv('barcos_ref.csv')
tabela_corr = (tabela.corr()[["Preco"]])

y = tabela['Preco']
x = tabela.drop('Preco', axis=1)

tabela_nova = pd.read_csv('novos_barcos.csv')


def treino():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    model_linear_regression = LinearRegression()
    model_forest_regression = RandomForestRegressor()

    model_linear_regression.fit(x_train, y_train)
    model_forest_regression.fit(x_train, y_train)

    prediction_linear = model_linear_regression.predict(x_test)
    prediction_forest = model_forest_regression.predict(x_test)

    print(metrics.r2_score(y_test, prediction_linear))
    print(metrics.r2_score(y_test, prediction_forest))

    new_prediction = model_forest_regression.predict(tabela_nova)
    return new_prediction





print(f'Previsao:\n {treino()}')
