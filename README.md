## Machine Learning

# 1. Redes Neuronales

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



# 3. Clasificacion

# Fundamentos de Clasificación: Del Clasificador de Bayes al Naive Bayes y Regresión Logística

## 1. Introducción y Definición del Problema

En el contexto de clasificación, buscamos predecir una etiqueta discreta \(y\) a partir de un vector de características \(x \in \mathbb{R}^d\). A diferencia de la regresión, donde la función de respuesta \(g(x)\) toma valores reales, en clasificación \(g(x)\) asigna una clase:

\[
g: \mathbb{R}^d \to \mathcal{P}
\]

donde \(\mathcal{P}\) es el conjunto finito de posibles clases. Normalmente, asumimos que \(|\mathcal{P}| < \infty\).

## 2. Función de Riesgo y Pérdida 0-1

Para evaluar un clasificador \(g\), utilizamos la **pérdida 0-1**, definida como:

\[
L(g(x), y) = \mathbb{I}(y \neq g(x)) = 
\begin{cases} 
1, & y \neq g(x) \\ 
0, & y = g(x) 
\end{cases}
\]

El **riesgo** asociado es la probabilidad de error de clasificación:

\[
R(g) := \mathbb{E}[\mathbb{I}(y \neq g(x))] = \mathbb{P}(y \neq g(x))
\]

Este riesgo se puede expresar integrando sobre la distribución de \(x\):

\[
R(g) = \int_{\mathbb{R}^d} \mathbb{P}(y \neq g(x) \mid x) f(x) \, dx
\]

Para un punto fijo \(x\), el clasificador óptimo minimiza la probabilidad de error condicional.

## 3. Clasificador de Bayes

El **clasificador de Bayes** es aquel que minimiza el riesgo para cada \(x\) de forma puntual. Para un problema con dos clases \(c_1, c_2\), se demuestra que la regla óptima es:

\[
g(x) = 
\begin{cases} 
c_1, & \text{si } \mathbb{P}(y = c_1 \mid x) \ge \mathbb{P}(y = c_2 \mid x) \\ 
c_2, & \text{en caso contrario}
\end{cases}
\]

De forma equivalente, para un conjunto de clases \(\mathcal{P}\):

\[
g(x) = \arg \max_{j \in \mathcal{P}} \mathbb{P}(y = j \mid x)
\]

Esta regla se conoce como el **clasificador de Bayes** y es óptimo en el sentido de minimizar la probabilidad de error.

### 3.1 Regla de decisión para dos clases

Para dos clases, la regla se puede reescribir como:

\[
g(x) = c_1 \quad \Longleftrightarrow \quad \mathbb{P}(y = c_1 \mid x) \ge \frac{1}{2}
\]

## 4. Método *Plug-in*: Estimación de las Probabilidades a Posteriori

En la práctica, las probabilidades \(\mathbb{P}(y = j \mid x)\) no son conocidas. El enfoque *plug-in* consiste en:

1. Estimar \(\hat{\mathbb{P}}(y = j \mid x)\) a partir de los datos.
2. Utilizar el clasificador:

\[
g(x) = \arg \max_{j \in \mathcal{P}} \hat{\mathbb{P}}(y = j \mid x)
\]

Para estimar \(\mathbb{P}(y = j \mid x)\) se pueden emplear diferentes modelos, entre ellos el **Naive Bayes** y la **regresión logística**.

## 5. Clasificador Naive Bayes

El clasificador Naive Bayes se basa en el **Teorema de Bayes** y en una suposición de independencia condicional:

\[
\mathbb{P}(y = c \mid x) = \frac{f(x \mid y = c) \, \mathbb{P}(y = c)}{\sum_{s \in \mathcal{P}} f(x \mid y = s) \, \mathbb{P}(y = s)}
\]

donde \(f(x \mid y = s)\) es la densidad condicional de \(x\) dada la clase \(s\).

La suposición *Naive* (ingenua) establece que, dado \(y = s\), las componentes de \(x\) son independientes:

\[
f(x \mid y = s) = \prod_{j=1}^{d} f(x_j \mid y = s)
\]

Aunque esta suposición es fuerte, suele conducir a clasificadores eficientes.

### 5.1 Estimación de las marginales \(\mathbb{P}(y = s)\)

Las probabilidades a priori se estiman mediante las frecuencias relativas en la muestra de entrenamiento:

\[
\hat{\mathbb{P}}(y = s) = \frac{|J_s|}{n}, \quad \text{donde } J_s = \{i: y_i = s\}
\]

### 5.2 Caso continuo: Distribución Gaussiana

Cuando las variables son continuas, una opción común es modelar cada \(f(x_j \mid y = s)\) como una distribución normal:

\[
x_j \mid y = s \sim \mathcal{N}(\mu_{js}, \sigma_{js}^2)
\]

Los parámetros se estiman por máxima verosimilitud:

\[
\hat{\mu}_{js} = \frac{1}{|J_s|} \sum_{i \in J_s} x_{j,i}, \quad 
\hat{\sigma}_{js}^2 = \frac{1}{|J_s|} \sum_{i \in J_s} (x_{j,i} - \hat{\mu}_{js})^2
\]

La densidad condicional estimada es:

\[
\hat{f}(x \mid y = s) = \prod_{j=1}^{d} \frac{1}{\sqrt{2\pi \hat{\sigma}_{js}^2}} \exp\left( -\frac{1}{2\hat{\sigma}_{js}^2} (x_j - \hat{\mu}_{js})^2 \right)
\]

### 5.3 Caso discreto: Distribución Multinomial

Si las variables son categóricas, se modela \(x_j \mid y = s\) como una distribución multinomial. En particular, para una variable \(x_j\) con \(K\) categorías, se tiene:

\[
x_j \mid y = s \sim \text{Multinomial}(1, \theta_{j,s})
\]

donde \(\theta_{j,s} \in \mathbb{R}^K\) es el vector de probabilidades por categoría, estimado por frecuencias relativas.

### 5.4 Problemas numéricos y solución logarítmica

El producto de muchas densidades (o probabilidades) puede resultar numéricamente inestable, tendiendo a cero. Para evitar esto, se trabaja en escala logarítmica. Dado que:

\[
\hat{\mathbb{P}}(y = c \mid x) \propto \hat{f}(x \mid y = c) \, \hat{\mathbb{P}}(y = c)
\]

la regla de clasificación se convierte en:

\[
g(x) = \arg\max_{c \in \mathcal{P}} \left( \log \hat{\mathbb{P}}(y = c) + \sum_{j=1}^{d} \log \hat{f}(x_j \mid y = c) \right)
\]

Este enfoque evita el subdesbordamiento numérico.

### 5.5 Generalización: *Flexible Bayes* y otras alternativas

Existen versiones más generales del Naive Bayes, conocidas como *Flexible Bayes*, que relajan la suposición de independencia. También se pueden utilizar métodos no paramétricos para estimar las densidades condicionales, como la estimación por núcleos (*Kernel Density Estimation*).

## 6. Regresión Logística: Otro Enfoque *Plug-in*

La regresión logística modela directamente \(\mathbb{P}(y = c \mid x)\) sin necesidad de estimar densidades conjuntas. Para dos clases (codificadas como \(y \in \{0,1\}\)), se supone:

\[
\mathbb{P}(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \sum_{j=1}^d \beta_j x_j)}}
\]

La estimación de los coeficientes \(\beta\) se realiza por **máxima verosimilitud**. Dada una muestra i.i.d. \(\{(x_k, y_k)\}_{k=1}^n\), la verosimilitud es:

\[
L(\beta) = \prod_{k=1}^n \left( \frac{1}{1 + e^{-(\beta_0 + \sum_{j=1}^d \beta_j x_{j,k})}} \right)^{y_k} \left( \frac{1}{1 + e^{\beta_0 + \sum_{j=1}^d \beta_j x_{j,k}}} \right)^{1 - y_k}
\]

Maximizar el logaritmo de esta verosimilitud proporciona los coeficientes estimados \(\hat{\beta}\).

### 6.1 Extensión a múltiples clases

Para más de dos clases, se pueden ajustar modelos logísticos multinomiales (softmax). Una estrategia práctica es ajustar una regresión logística binaria para cada clase \(c\) definiendo \(Z_c = \mathbb{I}(y = c)\) y luego combinar las probabilidades estimadas:

\[
g(x) = \arg\max_{c \in \mathcal{P}} \hat{\mathbb{P}}(y = c \mid x)
\]

## 7. Resumen y Consideraciones Finales

1. **Clasificador de Bayes**: regla óptima basada en las probabilidades a posteriori.
2. **Método *Plug-in***: estima dichas probabilidades a partir de los datos.
3. **Naive Bayes**: estimación paramétrica de las densidades condicionales bajo independencia.
   - Para variables continuas: modelo Gaussiano.
   - Para variables discretas: modelo Multinomial.
   - Se recomienda trabajar en escala logarítmica para estabilidad numérica.
4. **Regresión Logística**: modela directamente la probabilidad a posteriori mediante una función sigmoidea (o softmax para múltiples clases).

Estos métodos constituyen la base teórica y práctica de muchos sistemas de clasificación supervisada.

---

**Nota:** Este documento ha sido elaborado a partir de las notas de clase del curso de Machine Learning Avanzado dictado por el PDr. Helder Rojas del Programa de Doctorado en cinesncias e Ingenieria estadistica de la Universidad Nacional de Ingenieria











