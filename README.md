## Machine Learning

#1. Redes Neuronales

## Veamos un Problema Simple (Regresión Lineal)

- $y_i$
- $x_i$
- $y$
- $x$

- Tenemos datos de entrenamiento $X = \{x_i^N\}$, $i=1,\ldots,N$ con salida correspondiente $Y = \{y^N\}$, $i=1,\ldots,N$
- Queremos encontrar los parámetros que predicen la salida $Y$ a partir de los datos $X$ de forma lineal:

  $$Y \approx w_0 + w_1 x_i$$


**Notaciones**:

- **Superíndice**: Índice del punto de datos en el conjunto de entrenamiento; $k = k^h$ punto de datos de entrenamiento
- **Subíndice**: Coordenada del punto de datos

  $$X_j^k = \text{coordenada 1 del punto de datos } k.$$

- Tenemos datos de entrenamiento

  $$X = \{X_j^k\},\quad k=1,\ldots,N$$

  con salida correspondiente

  $$Y = \{y^k\},\quad k=1,\ldots,N$$

- Queremos encontrar los parámetros que predicen la salida $Y$ a partir de los datos $X$ de forma lineal:

  $$y^k \approx w_0 + w_i x_i^k$$

Es conveniente definir un atributo adicional "falso" para los datos de entrada:

$$x_0 = 1$$

Queremos encontrar los parámetros que predicen la salida $Y$ a partir de los datos $X$ de forma lineal:

$$y^k \approx w_0 x_0^k + w_i x_i^k$$

## Notaciones Más Convenientes

- Vector de atributos para cada punto de datos de entrenamiento:

  $$\mathbf{x}^k = [x_0^k, \ldots, x_M^k]$$

- Buscamos un vector de parámetros: $\mathbf{w} = [w_0, \ldots, w_M]$

- Tal que tenemos una relación lineal entre la predicción $Y$ y los atributos $X$:

  $$y^k \approx w_0 x_0^k + w_1 x_1^k + \cdots + w_M x_M^k = \sum_{i=0}^{M} w_i x_i^k = \mathbf{w} \cdot \mathbf{x}^k$$

Por definición: El producto punto entre vectores $\mathbf{w}$ y $\mathbf{x}^k$ es:

$$\mathbf{w} \cdot \mathbf{x}^k = \sum_{i=0}^{M} w_i x_i^k$$

## Red Neuronal: Perceptrón Lineal

- Valores de atributos de entrada
- Predicción de salida

Nota: Esta unidad de entrada corresponde al atributo "falso" $x_0 = 1$. Llamado el **sesgo (bias)**.

## Conexión con peso

**Unidades de Entrada**

*Problema de Aprendizaje de Redes Neuronales*: Ajustar los pesos de conexión para que la red genere la predicción correcta en los datos de entrenamiento.

## Regresión Lineal: Descenso de Gradiente

Buscamos un vector de parámetros $w = [w_0, \ldots, w_M]$ que minimice el error entre la predicción $Y$ y los datos $X$:

$$E = \sum_{k=1}^{N} (y^k - (w_0 x_0^k + w_1 x_1^k + \cdots + w_M x_M^k))^2$$

$$= \sum_{k=1}^{N} (y^k - \mathbf{w} \cdot \mathbf{x}^k)^2 = \sum_{k=1}^{N} \delta_k^2 \quad \text{donde } \delta_k = y^k - \mathbf{w} \cdot \mathbf{x}^k$$

## Descenso de Gradiente

El mínimo de $E$ se alcanza cuando las derivadas con respecto a cada uno de los parámetros $w_i$ es cero:

$$\frac{\partial E}{\partial w_i} = -2 \sum_{k=1}^{N} (y^k - \mathbf{w} \cdot \mathbf{x}^k) x_i^k = -2 \sum_{k=1}^{N} \delta_k x_i^k$$

## Regla de Actualización del Descenso de Gradiente

Moverse en la dirección opuesta a la dirección del gradiente:

$$w_i \leftarrow w_i - \alpha \frac{\partial E}{\partial w_i}$$

## Entrenamiento del Perceptrón

1. Calcular error:

   $$\delta_k \leftarrow y^k - \mathbf{w} \cdot \mathbf{x}^k$$

2. Actualizar pesos de la RN:

   $$w_i \leftarrow w_i + \alpha \delta_k x_i^k$$

## Parámetro de Tasa de Aprendizaje

$\alpha$ es la **tasa de aprendizaje**.

- $\alpha$ demasiado pequeña → converge lentamente.
- $\alpha$ demasiado grande → puede oscilar alrededor del mínimo.

## Un Problema de Clasificación Simple

Supongamos que tenemos un atributo $x_1$ y los datos están en dos clases.

Definimos la salida $y$ como:

$$y = 
\begin{cases} 
1 & \text{si en clase verde} \\ 
0 & \text{si en clase roja}
\end{cases}$$

Usamos la función sigmoide para obtener una versión suave:

$$\sigma(t) = \frac{1}{1 + e^{-t}}$$

Predicción: $y = \sigma(\mathbf{w} \cdot \mathbf{x})$

## Generalización a M Atributos

Una separación lineal está parametrizada como:

$$\sum_{i=0}^{M} w_i x_i = \mathbf{w} \cdot \mathbf{x} = 0$$

La red de una capa para clasificación es:

$$\hat{y} = \sigma(\mathbf{w} \cdot \mathbf{x})$$

## Entrenamiento con Sigmoide

1. Calcular error:

   $$\delta_k \leftarrow y^k - \sigma(\mathbf{w} \cdot \mathbf{x}^k)$$

2. Actualizar pesos:

   $$w_i \leftarrow w_i + \alpha \delta_k x_i^k \sigma'(\mathbf{w} \cdot \mathbf{x}^k)$$

## Redes Multicapa (Backpropagation)

Las redes multicapa pueden representar límites de decisión arbitrarios (no solo lineales).

El entrenamiento se realiza mediante **retropropagación** (backpropagation), que propaga los errores hacia atrás para actualizar todos los pesos.

## Resumen

Las redes neuronales se usan para:

- **Regresión**: aproximar $y$ como función continua de $x$
- **Clasificación**: predecir clases discretas

**Conceptos clave**:
- Función sigmoide (o activaciones no lineales)
- Descenso de gradiente
- Retropropagación
- Validación para evitar sobreajuste

**Ventajas**: Marco simple y poderoso.  
**Desventajas**: Muchos hiperparámetros, riesgo de sobreajuste, entrenamiento puede ser lento y caer en mínimos locales.



# **2. Arboles**


# Árboles de Regresión (CART)

## Definición básica

$$
f: x_i \mapsto y_i
$$

$$
p(x,y)
$$

$$
f(x) = E(Y \mid x)
$$

$$
\hat{f}(x) = g(x)
$$

Los árboles de regresión (Regression Trees, RT) son una metodología **no paramétrica** que permite obtener modelos altamente interpretables mediante particiones recursivas del espacio de covariables.


## Estructura de un Árbol Binario

Un árbol divide el espacio $\mathbb{R}^d$ en regiones mediante reglas binarias.

Ejemplo conceptual:

- Nodo raíz → condición
- Nodos internos → decisiones
- Hojas → predicciones



## Ejemplo de Regresión

**Objetivo:** Predecir ingreso en función de:

- Edad
- Años de estudio

Reglas:

- Si Edad ≤ 18 → ingreso ≈ 500  
- Si Edad > 18 y Estudios ≤ 10 → ingreso ≈ 1500  
- Si Edad > 18 y Estudios > 10 → ingreso ≈ 3000  



## Definición Formal

Un árbol de decisión construye una partición del espacio en regiones disjuntas:

$$
R_1, R_2, \ldots, R_J
$$

Predicción:

$$
\hat{g}(\mathbf{x}) =
\frac{1}{|R_k|}
\sum_{\mathbf{x}_i \in R_k} Y_i
$$


Es decir, se usa el **promedio de la región**.



## Construcción del Árbol

Se realiza en dos etapas:

1. Crecer un árbol grande
2. Podar el árbol (evitar overfitting)



## Formulación Matemática

Dataset:

$$
D_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n
$$

Función objetivo:

$$
f(\mathbf{x}) = E(Y \mid \mathbf{x})
$$

Estimación:

$$
g(\mathbf{x}) = \hat{f}(\mathbf{x})
$$

---

## Riesgo

### Riesgo esperado

$$
R(f) = E[(Y - f(\mathbf{x}))^2]
$$

### Riesgo empírico

$$
\hat{R}(f) =
\frac{1}{n} \sum_{i=1}^n (y_i - f(\mathbf{x}_i))^2
$$

---

## Partición del espacio

El modelo final:

$$
\hat{f}(\mathbf{x}) =
\sum_{k=1}^{M} C_k \cdot 1_{\{\mathbf{x} \in R_k\}}
$$

donde:

- $C_k$ = promedio en la región $R_k$

### Función indicadora

$$
1_{\{\mathbf{x} \in R_k\}} =
\begin{cases}
1 & \text{si } \mathbf{x} \in R_k \\
0 & \text{si no}
\end{cases}
$$

---

## Criterio de División (Split)

Para un nodo $s$:

### Error (SSE)

$$
SSE(s) =
\sum_{i \in S} (Y_i - \overline{Y}_s)^2
$$

---

## División del nodo

Se divide en:

- $S_L$ (izquierda)
- $S_R$ (derecha)

Ganancia:

$$
\Delta(j,t) =
SSE(s) - (SSE(S_L) + SSE(S_R))
$$



## Mejor split


$$
(j^{(k)}, t^{(k)}) =
\arg\max_{j,t} \Delta(j,t)
$$



## Algoritmo CART

1. **Inicialización**
   $$
   S = D_n
   $$

2. **Split recursivo**
   - Evaluar variables $X_j$
   - Evaluar umbrales $t$
   - Calcular $\Delta(j,t)$
   - Elegir el mejor split

3. **Criterios de parada**
   - $|S| < n_{min}$
   - $\Delta < \epsilon$
   - profundidad > $d_{max}$



## Predicción final

En hojas:

$$
\hat{f}(\mathbf{x}) = \overline{Y}_k
\quad \text{si } \mathbf{x} \in R_k
$$

Forma equivalente:

$$
\hat{f}(\mathbf{x}) =
\sum_{k=1}^{M} \overline{Y}_k
1_{\{\mathbf{x} \in R_k\}}
$$



## Notas importantes

- Modelo no paramétrico
- Alta interpretabilidad
- Puede sobreajustar → usar poda
- Base de métodos como:
  - Random Forest
  - Gradient Boosting












