# Predicción del precio de la vivienda en Pakistán

Práctica de Aprendizaje Automático del Máster en Ciencia de Datos e IA de CUNEF (curso 2026-2027).

**Repositorio:** https://github.com/rmataixser/Practica-Machine-Learning

**Autores:**

| Nombre | Correo CUNEF |
| --- | --- |
| David Martín Montalbán | d.martinmontalban@cunef.edu |
| Ricardo Mataix Serrano | ricardo.mataix@cunef.edu |
| Luis Prados Puertas | luis.prados@cunef.edu |
| Juan Pablo Dávila Martínez | juanpablo.davila@cunef.edu |
---

## 1. El problema

Queremos estimar a qué precio se vende una vivienda en las cinco grandes ciudades de Pakistán (Karachi, Lahore, Islamabad, Rawalpindi y Faisalabad) a partir de sus características: tipo, superficie, habitaciones, baños y ubicación.

Hoy, quien compra o vende tiene poca referencia. El precio lo marcan el propietario o el agente, y la forma habitual de contrastarlo es mirar anuncios parecidos en un portal o pagar una tasación. La tasación es lenta, cuesta dinero y depende de quién la haga. Comparar anuncios a mano funciona mal en cuanto la vivienda no es "típica", y los precios medios por zona no tienen en cuenta ni el tipo ni el tamaño de la vivienda.

Un modelo que dé un precio estimado en segundos le sirve a varias partes:

- al comprador, para saber si lo que le piden es razonable;
- al vendedor o al agente, para fijar un precio de salida;
- al propio portal, para avisar de anuncios con precios fuera de mercado.

## 2. Los datos

### Origen

Usamos los anuncios del portal inmobiliario [zameen.com](https://www.zameen.com), el más grande de Pakistán, descargados del conjunto [Zameen.com Property Data Pakistan](https://www.kaggle.com/datasets/huzzefakhan/zameencom-property-data-pakistan) de Kaggle.

El archivo es `Property.csv`: **191.393 anuncios** publicados o actualizados entre agosto de 2018 y agosto de 2019, con 17 columnas: tipo de vivienda, precio en rupias, zona y subzona, ciudad, provincia, coordenadas, baños, habitaciones, superficie, venta o alquiler, fecha, agencia y agente.

Al principio trabajamos con `Entities.csv`, una versión del mismo conjunto que circula también en internet. Al revisarla vimos que estaba mal procesada. La superficie original viene como texto en las unidades locales (`"4.8 Marla"`, `"1 Kanal"`), y al pasarla a pies cuadrados se perdió el punto decimal: `"4.8 Marla"` se convirtió como si fuera 48 marla. Así, **59.679 anuncios (el 35 %) tenían la superficie multiplicada por 10**. Además le faltaban 22.947 anuncios. Por eso partimos del archivo original y hacemos nosotros la conversión.

### Ventajas

- **Mucho volumen:** casi 112.000 anuncios de venta distintos, y unos 65.000 tras la limpieza.
- **Cinco ciudades** con mercados muy distintos, desde Lahore y Karachi hasta Faisalabad.
- **Ubicación en tres niveles:** zona, subzona y coordenadas. Con ellas calculamos la distancia al centro de cada ciudad.
- **Datos reales de un portal en uso,** no simulados.

### Desventajas

- **Son precios de anuncio, no de venta.** El precio final puede ser otro.
- **Muchos datos ausentes escondidos como ceros.** El 23,6 % de los anuncios de venta tiene 0 baños y el 11,2 %, 0 habitaciones. No es al azar: en Faisalabad faltan los baños en el 65 % de los anuncios, y en Lahore en el 18 %. Como decidimos eliminar esos anuncios, Faisalabad queda con muy pocos datos (menos del 3 %).
- **Anuncios repetidos.** Los agentes publican la misma vivienda varias veces, a veces con pequeños cambios en la descripción. Hay que tenerlo en cuenta al separar train y test.
- **Las coordenadas son de la subzona,** no de cada vivienda: hay unos 8.800 puntos distintos para todo el conjunto.
- **`date_added` es la fecha de la última actualización,** no la de publicación. Casi todas caen en junio y julio de 2019, así que no sirve para estudiar la evolución del precio.
- **Solo cubre un año, 2018-2019.** El modelo describe el mercado de entonces.
- **Los precios están en rupias.** Los pasamos a euros con el cambio del 18/07/2019 (1 € = 179,61 PKR) para que se entiendan mejor. Al ser una constante, no afecta al modelo.

## 3. Planteamiento como problema de Aprendizaje Automático

- **Tipo de problema:** aprendizaje supervisado, **regresión**.
- **Variable objetivo:** `price`, el precio de venta en euros. Lo modelamos en escala logarítmica porque está muy sesgado a la derecha (asimetría de 8,7, frente a −0,3 en logaritmos). Además, en logaritmos el error se interpreta en términos relativos, que es lo que importa en un precio.
- **Anuncios que usamos:** solo los de venta. El alquiler es otro problema, con precios mensuales en otra escala.
- **Vector de atributos:**

| Atributo | Tipo | Descripción |
| --- | --- | --- |
| `property_type` | Categórico | `House` (casa) o `Flat` (piso; agrupa también plantas, áticos y habitaciones) |
| `city` | Categórico | Una de las 5 ciudades |
| `location` | Categórico | Zona (unos 1.400 valores distintos) |
| `location_id` | Categórico | Subzona dentro de la zona (unos 3.800 valores) |
| `latitude`, `longitude` | Numérico | Coordenadas de la subzona |
| `dist_centro_km` | Numérico | Distancia al centro de la ciudad (fórmula de Haversine) |
| `Total_Area` | Numérico | Superficie en m² |
| `bedrooms` | Numérico | Habitaciones (de 1 a 12) |
| `baths` | Numérico | Baños (de 1 a 12) |
| `date_added` | Fecha | Fecha de actualización del anuncio (no la usamos en el modelo) |

## 4. Enfoque

1. **Descripción de los datos:** tipos, ausentes, duplicados, valores imposibles.
2. **Limpieza con reglas fijas:** solo ventas, conversión de unidades (la superficie a m² desde el texto original y el precio a euros), eliminación de los anuncios con 0 baños o 0 habitaciones (son datos sin rellenar), límites de superficie por tipo de vivienda (pisos hasta 600 m² y casas por debajo de 5.000 m²), coordenadas fuera de su ciudad y anuncios repetidos. Como son reglas que miran cada fila por separado, se pueden aplicar antes de separar train y test sin que haya fuga de información. Pasamos de 191.393 anuncios a 64.722 viviendas en venta.
3. **Partición train/test (80/20)**, agrupando los anuncios que pueden ser la misma vivienda (misma ciudad, zona, tipo, superficie y precio) y estratificando por ciudad. Con una partición aleatoria normal, casi la mitad del test (un 48 %) tendría un "gemelo" en el train y la evaluación saldría inflada.
4. **Análisis exploratorio sobre el train:** distribuciones, relaciones con el precio, diferencias entre ciudades y tipos, y atípicos, con las conclusiones y acciones que se derivan de cada análisis.
5. **Preparación para el modelado,** ajustada solo con el train: codificación de la zona y la subzona con el precio medio de cada una, *one-hot* de ciudad y tipo, nuevas variables (superficie y baños por habitación, logaritmo de la superficie) y escalado, todo dentro de un `Pipeline` de scikit-learn.
6. **Modelado:** empezaremos por un modelo de referencia (regresión lineal) y lo compararemos con modelos regularizados y de árboles (*random forest* y *gradient boosting*), ajustando hiperparámetros con validación cruzada agrupada. Mediremos el error con RMSE y MAE sobre el logaritmo del precio, y además con el error porcentual, que es fácil de interpretar en un precio.

## 5. Estructura del repositorio

```
Practica-Machine-Learning/
├── data/                         # no se sube al repositorio
│   ├── bronze/Property.csv       # datos originales
│   └── silver/                   # salida.csv (datos limpios), train.csv y test.csv
├── notebooks/
│   ├── 00_descripcion_datos.ipynb
│   ├── 01_limpieza.ipynb
│   ├── 02_particion_train_test.ipynb
│   └── 03_EDA.ipynb
├── src/
│   └── graficos.py               # funciones de gráficos para el EDA
├── html/                         # notebooks exportados a HTML
├── requirements.txt
└── README.md
```

## 6. Cómo reproducirlo

1. Clonar el repositorio:
   ```
   git clone https://github.com/rmataixser/Practica-Machine-Learning.git
   ```
2. Descargar `Property.csv` de [Kaggle](https://www.kaggle.com/datasets/huzzefakhan/zameencom-property-data-pakistan) y guardarlo en `data/bronze/`.
3. Instalar las librerías:
   ```
   pip install -r requirements.txt
   ```
4. Ejecutar los notebooks en orden, de `00` a `03`.