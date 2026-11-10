# Plan: Olist ML (3 misiones)

Misiones elegidas (de más fácil a más difícil):

1. 🚚 **Predicción de Entregas** (regresión)
2. 🛍️ **Recomendaciones con K-Means + PCA** (no supervisado)
3. 🧠 **Análisis de Sentimientos** (clasificación de texto)

Principio: **simple y legible**. Un notebook, `pd.read_csv` y nada de frameworks extra.

---

## Estructura del proyecto

```
C3_The_hatch/
├── data/                      # CSVs del dataset de Olist (Kaggle), tal cual vienen
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   └── product_category_name_translation.csv
├── olist_ml.ipynb             # el entregable
└── requirements.txt           # pandas, numpy, scikit-learn, matplotlib, seaborn
```

> `olist_geolocation_dataset.csv` y `olist_order_payments_dataset.csv` **no se usan**.

## Estructura del notebook

| Sección | Contenido |
|---|---|
| 0. Setup | Imports, `DATA = "data/"`, carga de los CSV, `RANDOM_STATE = 42` |
| 0.1 Helper | Una sola función `comparar_modelos(modelos, X, y, scoring)`: corre `cross_validate(..., return_train_score=True)` y devuelve un DataFrame con la media de train y test por modelo y el **gap** (train − test). Se usa en las misiones 1 y 3. |
| 1. Entregas | ver abajo |
| 2. Recomendaciones | ver abajo |
| 3. Sentimientos | ver abajo |
| 4. Conclusiones | Tabla resumen: misión, mejor modelo, métrica y por qué. Más 3 o 4 bullets de negocio. |

Cada misión sigue el mismo esqueleto corto:
**Datos → Features → Pipeline → Comparación (CV) → Overfitting → Visualización → Interpretación.**

---

## Misión 1: Predicción de Entregas 🚚

**Target:** `dias_entrega = order_delivered_customer_date − order_purchase_timestamp` (en días).

**Datos:**
- `orders`: filtrar `order_status == "delivered"` y descartar fechas nulas.
- `order_items` agregado por `order_id`: `n_items`, `precio_total`, `flete_total` y `seller_id` (el primero).
- `products`: `product_weight_g` y `product_category_name` (del primer ítem).
- `customers` → `customer_state`; `sellers` → `seller_state`.

**Features:**
| Pedido del enunciado | Feature |
|---|---|
| Ubicación | `customer_state`, `seller_state`, `mismo_estado` (0/1) |
| Categoría | `product_category_name` (top 15 + "otros") |
| Historial logístico | `flete_total`, `peso`, `n_items` |
| Volumen de pedidos | `pedidos_semana`: cantidad de pedidos en la semana de compra |
| Temporal | `mes`, `dia_semana` de la compra |

**Limpieza mínima:** quitar entregas de más de 60 días (outliers) y documentar cuántas filas se pierden.

**Pipeline:** `ColumnTransformer` (OneHotEncoder para categóricas, passthrough para numéricas) → modelo. Los árboles no necesitan escalado.

**Modelos:**
- `DecisionTreeRegressor` (sin límite de profundidad, para *mostrar* el overfitting, y otro con `max_depth=8`)
- `RandomForestRegressor(n_estimators=100)`
- `HistGradientBoostingRegressor` (rápido con ~95k filas)

**Métricas:** MAE (principal, se interpreta como "días de error"), RMSE y R².

**Overfitting:** tabla con train vs test de `comparar_modelos`. El árbol sin límite va a tener R² de train ≈ 1, y ese es el ejemplo didáctico.

**Visualizaciones:**
1. Histograma de `dias_entrega`.
2. Barras de MAE train vs test por modelo.
3. Scatter real vs predicho del mejor modelo.
4. Top 10 feature importances.

---

## Misión 2: Recomendaciones (K-Means + PCA) 🛍️

**Enfoque:** agrupar **productos** y recomendar los más vendidos del mismo cluster. Los clientes casi nunca repiten compra, así que clusterizar clientes no aporta.

**Datos (una fila por producto):**
- `products`: peso, dimensiones (largo, alto, ancho), `product_photos_qty` y categoría.
- `order_items` agregado por `product_id`: `precio_medio`, `flete_medio` y `n_ventas`.
- `reviews` (vía `orders` + `order_items`): `score_medio`.
- Filtrar productos con `n_ventas >= 3` para reducir ruido.

**Pipeline:** `ColumnTransformer` (`StandardScaler` para numéricas y `OneHotEncoder` para la categoría, top 15) → `PCA` → `KMeans`.

**"Comparación de modelos" (adaptada a no supervisado):**
- Elegir **k**: método del codo (inercia) + `silhouette_score` para k = 2..10.
- **Con PCA vs sin PCA:** comparar silhouette.
- PCA: gráfico de varianza explicada acumulada para elegir n componentes (~90%).

**Overfitting / estabilidad:** entrenar K-Means con el 80% de los productos, predecir el 20% restante y comparar el silhouette de ambos. Si se parecen, los clusters generalizan.

**Recomendador (una función de 5 líneas):**
`recomendar(product_id, n=5)` → busca su cluster y devuelve los `n` productos más vendidos de ese cluster, excluyéndolo a él.

**Visualizaciones:**
1. Codo + silhouette.
2. Varianza explicada del PCA.
3. Scatter PC1 vs PC2 coloreado por cluster.
4. Tabla "perfil de cluster": medias de cada feature y categoría dominante.

**Interpretación:** ponerle nombre a cada cluster (ej. "productos pesados y caros", "baratos y populares") y mostrar 2 o 3 ejemplos de recomendación.

---

## Misión 3: Análisis de Sentimientos 🧠

**Datos:** `olist_order_reviews_dataset.csv`.
- Quedarse con las filas que tienen `review_comment_message` (~40k).
- **Etiqueta:** `review_score` 1–2 → `negativo` (0), 4–5 → `positivo` (1). Se **descarta el 3** (ambiguo).
- Reportar el balance de clases (hay más positivos que negativos).

**Limpieza mínima:** minúsculas y quitar números y signos. El resto lo hace `TfidfVectorizer(strip_accents="unicode", min_df=5, ngram_range=(1,2))`.
Stopwords en portugués: una lista corta escrita a mano en el notebook (sin depender de nltk).

**Pipeline:** `TfidfVectorizer` → modelo.

**Modelos:**
- `MultinomialNB`
- `LinearSVC`
- `LogisticRegression` (baseline de referencia)

Todos con `class_weight="balanced"` donde aplique.

**Métricas:** F1 (principal, por el desbalance), precision, recall y `classification_report` del mejor modelo.

**Overfitting:** `comparar_modelos` con train vs test. Opcional: variar `C` en LinearSVC (0.01, 0.1, 1, 10) y graficar train/test (curva de validación simple).

**Patrones recurrentes (el extra):**
- Top 15 palabras/bigramas con más peso positivo y negativo según los coeficientes de LinearSVC.
- Se espera que aparezcan cosas como "não recebi", "atraso" o "chegou rápido", que conectan con la misión de entregas.

**Visualizaciones:**
1. Distribución de `review_score`.
2. Barras F1 train vs test por modelo.
3. Matriz de confusión del mejor modelo.
4. Barras horizontales de los términos más negativos y más positivos.

---

## Fuera de alcance (a propósito)

- Nada de GridSearch grandes: como mucho 1 parámetro por modelo y a mano.
- Ni SMOTE ni librerías extra (xgboost, nltk, spaCy).
- Sin geolocalización ni distancias reales.
- Sin módulos `.py` aparte: todo en el notebook.

## Orden de trabajo

1. Bajar el dataset a `data/` y crear `requirements.txt`.
2. Sección 0 (setup + helper).
3. Misión 1, que es la base y deja probado el helper.
4. Misión 3, que reutiliza el helper.
5. Misión 2.
6. Conclusiones.
