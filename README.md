# 🐦🐇 Clasificador de Aves y Conejos con CNN

Clasificador binario de imágenes que distingue entre **aves** y **conejos**, desarrollado con **TensorFlow/Keras**. El proyecto compara una red densa, una CNN y una CNN con Dropout, y exporta la CNN seleccionada a **TensorFlow.js** para su uso en una página web.

> **Estado:** prototipo académico. Los resultados son prometedores, pero el conjunto de datos es pequeño y la evaluación externa muestra que todavía hay margen de mejora.

## 📌 Resultados principales

| Modelo | `val_accuracy` | `val_loss` | Lectura |
|---|---:|---:|---|
| Red densa | 0.655 (65.5 %) | 0.639 | Rendimiento inferior a las CNN |
| CNN | **0.891 (89.1 %)** | 0.450 | Mayor accuracy de validación |
| CNN + Dropout | 0.873 (87.3 %) | **0.277** | Menor loss de validación |

La **CNN sin Dropout** (`modeloCNN_AD`) fue la seleccionada para exportación. La CNN + Dropout obtuvo una pérdida menor, aunque su accuracy fue 1.8 puntos porcentuales inferior. Con solo 55 imágenes de validación, esta diferencia no debe considerarse concluyente.

### Evaluación de la CNN exportada

**Matriz de confusión** en 55 imágenes de validación:

| Clase real | Predicha: Ave | Predicha: Conejo |
|---|---:|---:|
| Ave | 22 | 5 |
| Conejo | 4 | 24 |

- **46/55 aciertos (83.6 %)** según los conteos de la matriz.
- **Prueba externa:** 18/24 aciertos (75 %).
- Aves: 9/12 (75 %).
- Conejos: 9/12 (75 %).

La matriz de confusión y `val_accuracy` pueden diferir porque las cifras se obtuvieron de evaluaciones o momentos distintos del entrenamiento. La prueba externa es una evaluación separada y no debe confundirse con la validación interna.

## 🎯 Objetivo

Comparar arquitecturas de redes neuronales para clasificar imágenes de dos clases:

- **Ave** → etiqueta `0`
- **Conejo** → etiqueta `1`

Además, preparar el modelo convolucional para ejecutarlo desde el navegador mediante TensorFlow.js.

## 🗂️ Datos utilizados

| Clase | Fuente | Imágenes disponibles | Imágenes usadas |
|---|---|---:|---:|
| Ave | Bird Species Dataset (Kaggle) | 811, distribuidas en 6 especies | 182 |
| Conejo | Conejos Dataset v1 (Roboflow, formato COCO) | 182 | 182 |
| **Total** | | | **364** |

Especies de aves: Emperor Penguin, Carmine Bee-eater, Flamingo, American Goldfinch, Downy Woodpecker y Barn Owl.

Del dataset de conejos se utilizaron las imágenes; las anotaciones JSON de COCO no son necesarias para esta tarea de clasificación. Se seleccionaron 182 aves para balancear las clases.

## ⚙️ Preprocesamiento

- Redimensionado a **100 × 100 píxeles**.
- Conversión a **escala de grises** (un canal).
- Normalización: píxeles divididos entre `255`, para obtener valores entre `0` y `1`.
- División indicada en el proyecto: **85 % entrenamiento (309 imágenes)** y **15 % validación (55 imágenes)**.
- Aumento de datos aplicado **solo al conjunto de entrenamiento**.

Configuración de aumento de datos:

```python
ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True
)
```

No se aplica volteo vertical, para evitar transformaciones poco realistas.

## 🧠 Modelos comparados

| Modelo | Arquitectura resumida |
|---|---|
| Red densa | `Flatten → Dense(150) → Dense(150) → Dense(1, sigmoid)` |
| CNN | `3 × [Conv2D(32/64/128, 3×3) + MaxPooling] → Flatten → Dense(100) → Dense(1, sigmoid)` |
| CNN + Dropout | Arquitectura convolucional con `Dropout(0.5)` y `Dense(250)` |

Configuración común reportada:

- Optimizador: `Adam`
- Función de pérdida: `binary_crossentropy`
- Métrica: `accuracy`
- Hasta 100 épocas
- `EarlyStopping(patience=10, restore_best_weights=True)`
- Seguimiento con TensorBoard

### Matriz de confusión

Si guardas la imagen en `docs/images/matriz-confusion.png`, puedes mostrarla directamente en GitHub:

![Matriz de confusión de la CNN](docs/images/matriz-confusion.png)

La matriz contiene 22 aves y 24 conejos correctamente clasificados, con 9 errores en total.

## 📈 Interpretación de los resultados

- **Red densa:** obtuvo menor accuracy y curvas variables. Al aplanar la imagen al inicio, no aprovecha de forma explícita las relaciones espaciales locales como una CNN.
- **CNN:** logró el mayor `val_accuracy` (89.1 %).
- **CNN + Dropout:** alcanzó el menor `val_loss` (0.277), aunque su accuracy fue ligeramente inferior a la CNN sin Dropout.
- **Fotos externas:** el 75 % de aciertos muestra que el desempeño puede disminuir ante imágenes de otros orígenes. Se observaron confusiones con palomas blancas sobre fondos claros y conejos negros.

Accuracy mide la proporción de aciertos; loss también considera la confianza de las predicciones. Por ello, el modelo con mayor accuracy no necesariamente tiene la menor loss.

## 🚀 Cómo ejecutar el proyecto

El notebook usado en el proyecto es `Aves_Conejos.ipynb` y se desarrolló en Google Colab.

1. Abre el notebook en Google Colab.
2. Sube los archivos de datos esperados por el notebook:
   - `archive_birds.zip`
   - `Conejos Dataset.v1-wonejos.coco.zip`
3. Ejecuta las celdas en orden para descomprimir, cargar, balancear y preprocesar los datos.
4. Ejecuta el entrenamiento y la evaluación de los tres modelos.
5. Revisa las curvas, la matriz de confusión y las pruebas externas.

Dependencias principales: `tensorflow`, `opencv-python`, `numpy`, `matplotlib` y `scikit-learn`.

> Los nombres de los archivos de datos deben coincidir con los que espera el notebook. Si se cambian, actualiza las rutas correspondientes.

## 🌐 Exportar a TensorFlow.js

El modelo seleccionado se guardó como `aves-conejos-cnn-ad.h5`. El flujo de exportación documentado utiliza `tensorflowjs` y `tf_keras`, reconstruyendo la arquitectura y copiando los pesos capa por capa para resolver incompatibilidades de versiones.

En Google Colab, el proyecto utilizó:

```python
!pip install -q tensorflowjs tf_keras
```

Después de reconstruir el modelo, se guarda en una carpeta de salida, por ejemplo `model_tfjs/`, que debe contener los archivos generados por el conversor. En el proyecto se reportaron:

```text
model_tfjs/
├── model.json
├── group1-shard1of2.bin
└── group1-shard2of2.bin
```

Todos estos archivos deben permanecer en la misma carpeta. Los comandos de conversión pueden requerir ajustes según las versiones instaladas de TensorFlow, Keras, TensorFlow.js y Python; conviene registrar esas versiones para reproducir el proceso.

## 🖥️ Uso en una aplicación web

Contrato de entrada/salida del modelo:

| Elemento | Especificación |
|---|---|
| Entrada | Imagen de 100 × 100, escala de grises, un canal |
| Normalización | Valores de píxel entre 0 y 1 |
| Salida | Número entre 0 y 1 |
| Interpretación | `< 0.5` → Ave; `≥ 0.5` → Conejo |

Ejemplo ilustrativo con TensorFlow.js:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script>
async function clasificar(imgElement, modelo) {
  const tensor = tf.tidy(() =>
    tf.browser.fromPixels(imgElement, 1)
      .resizeBilinear([100, 100])
      .toFloat()
      .div(255)
      .expandDims(0)
  );

  try {
    const p = (await modelo.predict(tensor).data())[0];
    return p < 0.5
      ? { clase: "Ave", probabilidad: 1 - p }
      : { clase: "Conejo", probabilidad: p };
  } finally {
    tensor.dispose();
  }
}
</script>
```

Carga el modelo una sola vez con:

```javascript
const modelo = await tf.loadLayersModel("model_tfjs/model.json");
```

**Importante:** verifica que la transformación a escala de grises del navegador coincida con la usada durante el entrenamiento. `tf.browser.fromPixels(imgElement, 1)` puede no producir exactamente los mismos valores que OpenCV. Sirve la página desde un servidor local (por ejemplo, `python -m http.server`) en lugar de abrir el HTML directamente, para permitir la carga de `model.json`.

## ▶️ Ejecutar la aplicación web localmente

Para abrir la aplicación en tu navegador, inicia un servidor local desde la carpeta donde se encuentra el archivo principal de la web (por ejemplo, `index.html`).

1. Abre una terminal.
2. Ubícate en la carpeta de la aplicación. Por ejemplo:

   ```bash
   cd ruta/a/la/carpeta-de-la-app
   ```

3. Inicia el servidor con Python:

   ```bash
   python -m http.server 8000
   ```

4. Mantén la terminal abierta mientras utilizas la aplicación.
5. Abre tu navegador y visita **http://localhost:8000**.
6. Si la aplicación está dentro de una subcarpeta, entra a su ruta desde el navegador.

Para detener el servidor, vuelve a la terminal y presiona `Ctrl + C`.

**Importante:** Python publica la carpeta desde la que ejecutaste el comando y sus subcarpetas. Asegúrate de que `index.html` y la carpeta del modelo (`model_tfjs/`, con `model.json` y los archivos `.bin`) estén en las rutas esperadas por el código. Si el puerto `8000` está ocupado, puedes iniciar el servidor con `python -m http.server 8001` y abrir **http://localhost:8001**.

## ⚠️ Limitaciones conocidas

- Solo 364 imágenes en el conjunto balanceado; la validación contiene 55.
- Representación limitada de especies y apariencias, especialmente conejos oscuros.
- La escala de grises y el tamaño 100 × 100 eliminan información visual.
- La validación interna puede ser optimista respecto a fotografías externas.
- No se reportan intervalos de confianza ni resultados promedio de múltiples ejecuciones.

## 🔭 Mejoras futuras

1. Ampliar la cantidad y diversidad de imágenes, incluyendo conejos negros/grises y aves de más especies.
2. Crear un conjunto de prueba independiente antes de ajustar hiperparámetros.
3. Evaluar imágenes a color y una resolución mayor, por ejemplo 150 × 150.
4. Probar *transfer learning* con MobileNetV2 u otra red preentrenada.
5. Reportar precision, recall, F1-score y métricas por clase.
6. Repetir entrenamientos con distintas semillas y reportar media y desviación estándar.
7. Revisar posibles duplicados y diferencias entre fuentes de datos.

## 🖼️ Figuras del proyecto

Para que las curvas y la matriz se visualicen en GitHub, crea `docs/images/` y añade allí las imágenes con nombres sencillos, por ejemplo:

```text
docs/images/
├── accuracy-cnn.png
├── loss-cnn.png
├── accuracy-denso.png
├── loss-denso.png
├── accuracy-cnn-dropout.png
├── loss-cnn-dropout.png
└── matriz-confusion.png
```

En el README, utiliza rutas relativas como `![Matriz de confusión](docs/images/matriz-confusion.png)`. GitHub no podrá mostrar imágenes que solo existan en tu computador.

## 📁 Estructura orientativa

```text
.
├── Aves_Conejos.ipynb
├── aves-conejos-cnn-ad.h5
├── model_tfjs/
│   ├── model.json
│   ├── group1-shard1of2.bin
│   └── group1-shard2of2.bin
└── README.md
```

La estructura anterior es orientativa: incluye los artefactos mencionados en el proyecto y debe ajustarse a los archivos que efectivamente se suban al repositorio.

## 📄 Licencia y atribución de datos

Antes de publicar el repositorio, revisa las licencias y condiciones de uso de ambos datasets y añade sus atribuciones o enlaces originales. Si todavía no has elegido una licencia para el código, puedes incorporar un archivo `LICENSE` con una licencia compatible con el uso que quieras permitir.

---

**Resumen:** las CNN ofrecen una mejora clara frente a la red densa en este experimento. La CNN fue seleccionada por su mayor accuracy de validación; la CNN + Dropout obtuvo menor loss. La evaluación externa (75 %) indica que ampliar la diversidad de datos y validar con un conjunto independiente son los siguientes pasos más importantes.
