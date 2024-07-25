# [overfitting, underfitting]
fig, axs = plt.subplots(5, 2, figsize=(15,25))
for ax, M in zip(axs.flatten(), range(1, 11)):
    w = get_best_coef(X, Y, M)
    y_pred_plot = model_predict(w, x_true_plot)
    ax.set_title(f'Polinomio grado {M}')
    ax.plot(x_true_plot, y_true_plot, label='$f(x) = sin(2 \pi x)$', color='green')
    ax.plot(x_true_plot, y_pred_plot, label=f'polinomio grado {M}', color='red')
    ax.scatter(X, Y, marker='o', color='blue', label='Samples',)
    ax.legend()
    ax.set_ylim(-1.2,1.2)
    
    
# [training vs test error]
training_err = []
test_err = []
for M in range(1,10):
    w = get_best_coef(X, Y, M) # con estos pesos estimo train y test
    y_pred = model_predict(w, X)
    training_err.append(rms_err(Y, y_pred))
    y_pred_test = model_predict(w, X_test)
    test_err.append(rms_err(Y_test, y_pred_test))

plt.figure(figsize=(12,5))
plt.plot(training_err, label='Training', marker='x')
plt.plot(test_err, label='Test', marker='o')
plt.ylim(0,10)
plt.ylabel('RMSErr')
plt.xlabel('Poly order')
plt.legend(); plt.show()

#[Datos - split]
# generar toy dataset en pkl
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(
    iris.data, 
    columns=iris.feature_names
    )

df['target'] = iris.target

# Map targets to target names
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}

df['species'] = df['target'].map(target_names)
df = df.rename(columns={"petal length (cm)":"petal_length","petal width (cm)":"petal_width"})
df = df[["petal_length", "petal_width", "species"]]
df.to_pickle('toy_dataset.pkl')