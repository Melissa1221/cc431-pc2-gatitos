# Gatitos UNI: ojos abiertos vs ojos cerrados

Práctica Calificada 2 de CC431 Computación Gráfica (UNI, 2026-2), sobre el repo base
[letras_uni](https://github.com/PeterTXS09/letras_uni).

**Pregunta:** si varias personas dibujan el mismo gatito una vez mirando y otra a ciegas,
¿puede una red neuronal saber cuándo se dibujó sin retroalimentación visual?

Es un problema de clasificación con dos clases: `abiertos` y `cerrados`.

## Integrantes

- Melissa Iman Noriega
- Ivett Mera Amado
- Benjamin Seminario
- Martin Centeno

## Cómo dibujar

```bash
python3.12 -m venv .venv          # sirve 3.11, 3.12 o 3.13; con 3.14 no instalan las versiones fijadas
source .venv/bin/activate         # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Abrir <http://127.0.0.1:5001>, escribir tu nombre y dibujar lo que pida la app.

**Desde una tablet o celular:** con la compu y la tablet en el mismo WiFi, corre
`flask --app main run --host 0.0.0.0 --port 5001` y en la tablet abre `http://<IP de la compu>:5001`
(en Mac la IP sale con `ipconfig getifaddr en0`). Si usas lápiz, la app ignora la palma apoyada.

**Protocolo (lo seguimos todos):**

1. La figura es siempre un **gatito**: cabeza, orejas, ojos, bigotes y lo que salga.
2. La app elige la clase que te toca para que termines con la misma cantidad de las dos.
   La meta es **125 con ojos abiertos y 125 con ojos cerrados por persona**.
3. **Ojos cerrados:** pon el cursor dentro del lienzo, cierra los ojos y dibuja. Ábrelos solo
   al terminar. No borres para corregir; *Borrar* es solo para un trazo que se salió por accidente.
4. Siempre con el mismo nombre de autor, en minúsculas y sin espacios (ej. `melissa`).

Cada dibujo se guarda como `dataset/<clase>/<autor>_<timestamp>.png` (280×280, fondo
transparente) y se agrega una fila a `dataset/meta/<autor>.csv` con la duración del dibujo
y el número de trazos.

**Para juntar los datos:** cada uno sube su carpeta `dataset/` al repo (`git add dataset`,
commit y push). Los nombres llevan el autor y la hora, así que no chocan.

## Rutas de la app

| Ruta | Qué hace |
|---|---|
| `/` | Lienzo para dibujar |
| `/stats` | Cuántos dibujos hay por autor y por clase |
| `/prepare?size=64` | Genera `X.npy` (imágenes size×size, canal alfa), `y.npy` (clase) y `autores.npy`. Cada dibujo se recorta a su contorno y se centra; con `&recorte=0` se deja el lienzo completo |
| `/X.npy`, `/y.npy`, `/autores.npy` | Descarga los arreglos |

`autores.npy` sirve para probar la red con una persona que no vio al entrenar: si todos los
dibujos de una persona están en el test, la red no puede acertar reconociendo el estilo de
quien dibujó.

## Evaluación por persona no vista

Usamos **leave-one-author-out**: hacemos un fold por cada autor. En cada fold, todos los dibujos
de una persona se reservan para test y nunca entran en train ni en validation; el modelo se
entrena con los otros autores. En el dataset actual los IDs son `melissa`, `ivett`, `benja` y
`martin`, así que son cuatro folds.

| Fold | Test | Train y validation |
|---|---|---|
| 1 | `melissa` | `ivett`, `benja`, `martin` |
| 2 | `ivett` | `melissa`, `benja`, `martin` |
| 3 | `benja` | `melissa`, `ivett`, `martin` |
| 4 | `martin` | `melissa`, `ivett`, `benja` |

**Justificación:** no elegimos a la persona que dibuja mejor o peor, porque eso podría sesgar la
evaluación. Probamos con cada persona una vez para ver si el resultado depende de quién quedó
fuera. La separación se hace por autor, no repartiendo PNGs al azar: de otro modo la red podría
reconocer el estilo de alguien que ya vio durante el entrenamiento.

La validación también usa solo los autores de train de ese fold. Los resultados de los cuatro
folds son una evaluación cruzada de estas personas; no equivalen a un test externo con una quinta
persona que la red nunca haya visto en ninguna etapa.

## Entrenamiento y métricas

La CNN mínima usa *Global Average Pooling* antes de la salida. El script repite cada fold con cinco
semillas y guarda predicciones, métricas y CAM por persona.

1. Con las dependencias de la app instaladas, inicia `python main.py`.
2. Abre <http://127.0.0.1:5001/prepare?size=64> para generar `X.npy`, `y.npy` y `autores.npy`.
   Vuelve a generar estos archivos después de agregar dibujos nuevos.
3. En otra terminal, instala PyTorch y entrena:

   ```bash
   python -m pip install -r requirements-ml.txt
   python train_minimal_cnn.py --leave-one-author-out
   ```

Los resultados quedan en `model_results/leave_one_author_out/`. `evaluation.json` incluye los
resultados de cada autor como test, las predicciones out-of-fold y el resumen entre folds y
semillas. Cada fold contiene checkpoints, predicciones, historial de entrenamiento y CAM.

Después de agregar o limpiar PNGs, vuelve a abrir `/prepare?size=64` para regenerar `X.npy`,
`y.npy` y `autores.npy`. El entrenamiento usa los PNGs presentes en las carpetas de clase; los
CSV de metadata no sustituyen esa revisión y pueden conservar filas de archivos borrados durante
la limpieza.

Este es un baseline para desbloquear la evaluación; no incluye todavía la comparación con el MLP.
