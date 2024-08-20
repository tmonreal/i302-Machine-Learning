# Ideas sobre Clase 3

Charla con Robbie:
- Dar features categóricas. 
- Dar normalización (es una transformación lineal. lo podes sacar del min y max o media y varianza). Esos coeficientes pueden viajar con el modelo o entregar el modelo desnormalizado. Solo es para mejorar el número de condición del problema numérico. No debería modificar en nada más. (puede que el número de cond siga siendo malo x cuestiones geométricas, pero ya no por eso). 
- Dar One hot encoder
    - Fuentes: Probabilistric Machine Learning. Section 1.5.3.1
- Explicar el pipeline de trabajo. Cuando hacer el preprocesamiento, etc
- Regularización va a estar dado.
- Explicar la varianza. R2. Hay diferentes: Varianza de los datos naturales, varianza con respecto a la media y varianza con respecto al modelo. Con algún ejemplo unidimensional. 

## Fuentes:
- Working with categorical data. Google cloud workshop: https://developers.google.com/machine-learning/crash-course/categorical-data 
    - Bueno para seguir la estructura.
    - Es básico, habría que desprender mas temas a partir de ahí
- Encoding de features categoricas https://medium.com/aiskunks/categorical-data-encoding-techniques-d6296697a40f


Titulo de la clase: Feature Engineering
Temas:
- Features aprendidos vs features engineered
- Operaciones tipicas de feature engineering:
    1- Handling missing values
    2- Scaling/Normalization
    3- Discretizacion
    4- Encoding de features categoricas (https://medium.com/aiskunks/categorical-data-encoding-techniques-d6296697a40f)
        4.1 Que es la data categorica?
        4.2 Que es data encoding y tipos (aca entra one hot encoding)
    5- Feature crossing
- Data Leakage
    1. Que es y Causas tipicas de data leakage
    2. Como detectarlo
- Engineering Good Features
    1. Feature importance

Si sobran 20 minutos, explicar la varianza. R2. Hay diferentes: Varianza de los datos naturales, varianza con respecto a la media y varianza con respecto al modelo. Con algún ejemplo unidimensional. 