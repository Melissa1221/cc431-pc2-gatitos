import os
import re
import csv
import time
import base64
import glob
from flask import Flask, request, redirect, send_file, jsonify
from skimage import io
from skimage.transform import resize
import numpy as np

app = Flask(__name__)

# Las dos clases del problema: el mismo gatito dibujado mirando y a ciegas
CLASES = ["abiertos", "cerrados"]
DATASET = "dataset"
META = os.path.join(DATASET, "meta")
META_CAMPOS = ["archivo", "autor", "clase", "duracion_ms", "trazos", "fecha"]

main_html = """
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gatitos UNI</title>
<style>
  body { font-family: sans-serif; }
  #myCanvas { touch-action: none; background: white; }
  .abiertos { color: #1a7f37; }
  .cerrados { color: #8a1538; }
  #instruccion { max-width: 420px; margin: 0 auto; }
  #progreso { color: #555; }
</style>
</head>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;
  var inicio = null;   // momento del primer trazo
  var fin = null;      // momento del ultimo trazo
  var trazos = 0;      // cuantas veces se apoyo el lapiz

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");

      var autor = localStorage.getItem('autor') || '';
      document.getElementById('autor').value = autor;
      if (autor) { siguienteClase(autor); }

      $('#autor').change(function () {
          var a = limpiarAutor($(this).val());
          $(this).val(a);
          localStorage.setItem('autor', a);
          if (a) { siguienteClase(a); }
      });

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          empezarTrazo();
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
  	    $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });

      // Soporte tactil, para dibujar desde el celular o una tablet
      var canvas = document.getElementById('myCanvas');
      canvas.addEventListener('touchstart', function (e) {
          e.preventDefault();
          var t = e.touches[0];
          mousePressed = true;
          empezarTrazo();
          Draw(t.pageX - $(canvas).offset().left, t.pageY - $(canvas).offset().top, false);
      });
      canvas.addEventListener('touchmove', function (e) {
          e.preventDefault();
          var t = e.touches[0];
          if (mousePressed) {
              Draw(t.pageX - $(canvas).offset().left, t.pageY - $(canvas).offset().top, true);
          }
      });
      canvas.addEventListener('touchend', function (e) {
          mousePressed = false;
      });
  }

  function limpiarAutor(a) {
      return a.trim().toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g, '').replace(/[^a-z0-9]/g, '');
  }

  // Pide al servidor la clase que le toca a este autor (la que tiene menos dibujos),
  // asi cada persona termina con la misma cantidad de gatitos abiertos y cerrados
  function siguienteClase(autor) {
      $.getJSON('/siguiente?autor=' + autor, function (r) {
          document.getElementById('numero').value = r.clase;
          var m = document.getElementById('mensaje');
          if (r.clase == 'abiertos') {
              m.innerHTML = 'Dibuja un gatito con los OJOS ABIERTOS';
              document.getElementById('instruccion').innerHTML = 'Dibuja normal, mirando el lienzo.';
          } else {
              m.innerHTML = 'Dibuja un gatito con los OJOS CERRADOS';
              document.getElementById('instruccion').innerHTML =
                  'Ubica el cursor en el lienzo, <b>cierra los ojos</b> y dibuja. ' +
                  'Abrelos solo cuando termines. No borres para corregir: solo si el trazo se salio por accidente.';
          }
          m.className = r.clase;
          document.getElementById('progreso').innerHTML =
              'Llevas ' + r.conteo.abiertos + ' con ojos abiertos y ' + r.conteo.cerrados + ' con ojos cerrados (meta: 125 y 125)';
      });
  }

  function empezarTrazo() {
      if (inicio === null) { inicio = Date.now(); }
      trazos += 1;
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
          fin = Date.now();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      // Use the identity matrix while clearing the canvas
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      inicio = null; fin = null; trazos = 0;
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
     if (!document.getElementById('autor').value) {
         alert('Escribe tu nombre antes de enviar');
         return false;
     }
     if (inicio === null || fin === null) {
         alert('El lienzo esta vacio');
         return false;
     }
     var canvas = document.getElementById('myCanvas');
     document.getElementById('myImage').value = canvas.toDataURL();
     document.getElementById('duracion').value = fin - inicio;
     document.getElementById('trazos').value = trazos;
     document.getElementById('autorForm').value = document.getElementById('autor').value;
     return true;
  }



</script>
<body onload="InitThis();">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" ></script>
    <div align="left">
      <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" width="120"/>
    </div>
    <div align="center">
        <p>Tu nombre: <input id="autor" type="text" placeholder="ej. melissa"></p>
        <h1 id="mensaje">Escribe tu nombre para empezar</h1>
        <p id="instruccion"></p>
        <canvas id="myCanvas" width="280" height="280" style="border:2px solid black"></canvas>
        <br/>
        <br/>
        <button onclick="javascript:clearArea();return false;">Borrar</button>
        <p id="progreso"></p>
    </div>
    <div align="center">
      <form method="post" action="upload" onsubmit="return prepareImg();"  enctype="multipart/form-data">
      <input id="numero" name="numero" type="hidden" value="">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input id="autorForm" name="autor" type="hidden" value="">
      <input id="duracion" name="duracion" type="hidden" value="">
      <input id="trazos" name="trazos" type="hidden" value="">
      <input id="bt_upload" type="submit" value="Enviar">
      </form>
    </div>
</body>
</html>

"""


def limpiar_autor(autor):
    return re.sub(r'[^a-z0-9]', '', (autor or '').lower())


def contar(autor):
    return {c: len(glob.glob('{}/{}/{}_*.png'.format(DATASET, c, autor))) for c in CLASES}


@app.route("/")
def main():
    return(main_html)


@app.route('/siguiente', methods=['GET'])
def siguiente():
    autor = limpiar_autor(request.args.get('autor'))
    conteo = contar(autor)
    menor = min(conteo.values())
    empatadas = [c for c in CLASES if conteo[c] == menor]
    clase = empatadas[np.random.randint(len(empatadas))]
    return jsonify(clase=clase, conteo=conteo)


@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        clase = request.form.get('numero')
        autor = limpiar_autor(request.form.get('autor'))
        if clase not in CLASES or not autor:
            raise ValueError('clase o autor invalido: {} / {}'.format(clase, autor))
        # El nombre lleva el autor: hace falta para probar la red con una persona que no vio
        nombre = '{}_{}.png'.format(autor, int(time.time() * 1000))
        with open(os.path.join(DATASET, clase, nombre), 'wb') as fh:
            fh.write(base64.b64decode(img_data))
        # Un CSV por autor, para que al juntar los datos de todos en git no haya conflictos
        ruta_meta = os.path.join(META, '{}.csv'.format(autor))
        nuevo = not os.path.exists(ruta_meta)
        with open(ruta_meta, 'a', newline='') as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(META_CAMPOS)
            w.writerow([nombre, autor, clase, request.form.get('duracion'),
                        request.form.get('trazos'), time.strftime('%Y-%m-%d %H:%M:%S')])
        print("Image uploaded", clase, nombre)
    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/", code=302)


@app.route('/stats', methods=['GET'])
def stats():
    filas = []
    for archivo in glob.glob('{}/*/*.png'.format(DATASET)):
        clase = os.path.basename(os.path.dirname(archivo))
        autor = os.path.basename(archivo).split('_')[0]
        filas.append((autor, clase))
    autores = sorted(set(a for a, _ in filas))
    tabla = {a: {c: sum(1 for x in filas if x == (a, c)) for c in CLASES} for a in autores}
    total = {c: sum(1 for _, x in filas if x == c) for c in CLASES}
    return jsonify(por_autor=tabla, total=total)


@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    # ?size=64 por defecto; el lienzo mide 280x280 y se reduce para entrenar
    size = int(request.args.get('size', 64))
    images = []
    labels = []
    autores = []
    for clase in CLASES:
      filelist = sorted(glob.glob('{}/{}/*.png'.format(DATASET, clase)))
      if not filelist:
        continue
      images_read = io.concatenate_images(io.imread_collection(filelist))
      # El canal alfa vale 255 donde hay trazo y 0 en el fondo transparente
      images_read = images_read[:, :, :, 3]
      images_read = np.array([resize(im, (size, size), anti_aliasing=True, preserve_range=True)
                              for im in images_read]).astype(np.uint8)
      images.append(images_read)
      labels.append(np.array([clase] * images_read.shape[0]))
      autores.append(np.array([os.path.basename(f).split('_')[0] for f in filelist]))
    images = np.vstack(images)
    labels = np.concatenate(labels)
    autores = np.concatenate(autores)
    np.save('X.npy', images)
    np.save('y.npy', labels)
    np.save('autores.npy', autores)
    return "OK! {} imagenes de {}x{}".format(images.shape[0], size, size)

@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('./X.npy')
@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file('./y.npy')
@app.route('/autores.npy', methods=['GET'])
def download_autores():
    return send_file('./autores.npy')


for c in CLASES:
    os.makedirs(os.path.join(DATASET, c), exist_ok=True)
os.makedirs(META, exist_ok=True)

if __name__ == "__main__":
    # 5001 y no 5000: en Mac el puerto 5000 lo ocupa AirPlay
    app.run(debug=True, port=int(os.environ.get("PORT", 5001)))
