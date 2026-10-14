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
| `/prepare?size=64` | Genera `X.npy` (imágenes size×size, canal alfa), `y.npy` (clase) y `autores.npy` |
| `/X.npy`, `/y.npy`, `/autores.npy` | Descarga los arreglos |

`autores.npy` sirve para probar la red con una persona que no vio al entrenar: si todos los
dibujos de una persona están en el test, la red no puede acertar reconociendo el estilo de
quien dibujó.

## Entrenamiento y métricas

Ver el notebook en `notebooks/` (pendiente).
