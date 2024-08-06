# Reunión de Machine Learning Dataset MELI 2do semestre 2024

- Armando dataset de mercado de alquiler de deptos en AMBA.
- Publicaciones de 2021-2022.
- Separamos trimestralmente.
- En lugar de dar longitud y latitud de cada vivenda, arman un polígono y los que están dentro del polígono les asignan el centroide de ese polígono. Agregan ruido a la coordenada gps para anonimizar la data.
- Los precios van a estar ajustados por inflación. Precio del alquiler en valores constantes. 
- Features nos dan todos los que tienen. Hay muchos N/A (no informado). No todos sirven. Ellos en general las descartan porque la imputación no les sirve. 

La idea es que cuando recibamos el bache nuevo, hagas:
1.  un estudio exploratorio de los datos como para verificar que estén bien. Ahi podes interactuar con Abigail, para hacerle preguntas o pedirle algo específico.

2. Una vez que estemos OK con los datos (features, polígonos, etc), armes modelos para verificar que el problema es abordable con las técnicas de ML que damos en el curso. 

3. Finalmente, tenemos que definir el Enunciado del PF (simil lo que hicimos este semestre), y también definir como armaremos el Test set, sobre el cual evaluaremos los modelos de los estudiantes.
