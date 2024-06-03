import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# Генерация синтетических данных
def generate_synthetic_data(n=100):
    np.random.seed(0)
    data = np.random.dirichlet(np.ones(4), size=n)
    return pd.DataFrame(data, columns=['series1', 'series2', 'series3', 'series4'])

# Разделение на тренировочный и тестовый наборы данных
def train_test_split(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Обучение модели VAR
def train_var_model(train_data, maxlags=15):
    model = VAR(train_data)
    model_fitted = model.fit(maxlags=maxlags, ic='aic')
    return model_fitted

# Прогнозирование
def make_forecast(model_fitted, steps=10):
    forecast = model_fitted.forecast(model_fitted.y, steps=steps)
    return pd.DataFrame(forecast, columns=model_fitted.names)

# Визуализация прогнозов
def plot_forecast(train_data, test_data, forecast):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(train_data.columns):
        plt.subplot(2, 2, i + 1)
        plt.plot(train_data.index, train_data[col], label='Train')
        plt.plot(test_data.index, test_data[col], label='Test')
        plt.plot(forecast.index, forecast[col], label='Forecast')
        plt.title(col)
        plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Генерация синтетических данных
    data = generate_synthetic_data()
    
    # Разделение данных
    train_data, test_data = train_test_split(data)
    
    # Обучение модели VAR
    model_fitted = train_var_model(train_data)
    
    # Прогнозирование
    forecast_steps = len(test_data)
    forecast = make_forecast(model_fitted, steps=forecast_steps)
    forecast.index = test_data.index
    
    # Визуализация
    plot_forecast(train_data, test_data, forecast)

if __name__ == "__main__":
    main()
