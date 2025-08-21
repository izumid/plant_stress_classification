from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

%load_ext cuml.accel

# Dados fictícios
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + np.random.randn(100) * 0.1

# Configuração do KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()
errors = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred))

print(f"Erro médio: {np.mean(errors):.4f}")