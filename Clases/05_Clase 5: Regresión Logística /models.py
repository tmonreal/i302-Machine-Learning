import numpy as np

class LinearRegression:
    def __init__(self, degree=1, basis_fn=None):
        self.degree = degree
        # Default to polynomial basis if none provided
        if basis_fn is None:
            self.basis_fn = lambda x, d: np.power(x, d).flatten()  # Ensure it returns a 1D array
        else:
            self.basis_fn = basis_fn
        self.coef_ = None
        self.intercept_ = None

    def _design_matrix(self, X):
        """
        Constructs the design matrix PHI using the basis function.
        X: input feature vector (1D or 2D)
        """
        X = X.flatten()  # Ensure X is a 1D array
        PHI = np.zeros((len(X), self.degree + 1))
        for i in range(self.degree + 1):
            PHI[:, i] = self.basis_fn(X, i)  # Basis function applied on 1D array
        return PHI

    def fit(self, X, Y):
        """
        Fits the model to the data using the normal equation.
        X: input feature vector (1D)
        Y: target vector
        """
        PHI = self._design_matrix(X)
        
        # Normal equation: w = (PHI^T * PHI)^-1 * PHI^T * Y
        PHI_T_PHI = np.dot(PHI.T, PHI)
        PHI_T_Y = np.dot(PHI.T, Y)
        PHI_T_PHI_inv = np.linalg.inv(PHI_T_PHI)
        
        # Get coefficients
        w = np.dot(PHI_T_PHI_inv, PHI_T_Y)
        
        self.coef_ = w[1:]  # Coefficients for X terms
        self.intercept_ = w[0]  # Intercept term
    
    def predict(self, X):
        """
        Predicts the target values for input X using the fitted model.
        X: input feature vector (1D)
        """
        PHI = self._design_matrix(X)
        return np.dot(PHI, np.concatenate(([self.intercept_], self.coef_)))

    def __str__(self):
        """
        Return a string representation showing the intercept and coefficients.
        """
        coef_str = ' + '.join([f'{round(c,3)} * x^{i+1}' for i, c in enumerate(self.coef_)])
        return f"ŷ(x) = {round(self.intercept_,3)} + {coef_str}"
    

class LogisticRegression:
    def __init__(self, threshold=0.5, max_iter=1000, learning_rate=0.01):
        """
        Constructor para la clase de regresión logística.
        threshold: umbral para clasificar como clase 1 (por defecto 0.5)
        max_iter: número máximo de iteraciones para el gradiente descendente
        learning_rate: tasa de aprendizaje para la optimización
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None
    
    def _sigmoid(self, z):
        """
        Función sigmoide para transformar una entrada en una probabilidad.
        z: valor escalar o array numpy
        """
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """
        Agrega una columna de unos al inicio de X para el término de intercepción (bias).
        X: matriz de características de entrada
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        """
        Ajusta el modelo a los datos usando gradiente descendente.
        X: matriz de características (n_samples, n_features)
        y: vector de etiquetas (n_samples,)
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        
        # Inicializar los coeficientes
        self.coef_ = np.zeros(X.shape[1])
        
        # Gradiente descendente
        for _ in range(self.max_iter):
            # Predecir probabilidad
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            # Gradiente
            gradient = np.dot(X.T, (y_hat - y)) / y.size
            # Actualizar los coeficientes
            self.coef_ -= self.learning_rate * gradient
        
        self.intercept_ = self.coef_[0] # El coeficiente de intercepción es el primer valor de coef_
        self.coef_ = self.coef_[1:]
    
    def predict_proba(self, X):
        """
        Predice las probabilidades para cada clase para las entradas X.
        X: matriz de características (n_samples, n_features)
        """
        X = self._add_intercept(X)
        # Calcular probabilidades para la clase positiva (y=1) y negativa (y=0)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T
    
    def predict(self, X):
        """
        Predice las clases (0 o 1) para las entradas X, utilizando el umbral.
        X: matriz de características (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)
    
    def __str__(self):
        """
        Devuelve una representación en string del modelo mostrando el intercepto y los coeficientes.
        """
        coef_str = ' + '.join([f'{round(c, 3)} * x{i+1}' for i, c in enumerate(self.coef_)])
        return f'ŷ(x) = {round(self.intercept_, 3)} + {coef_str}'