# Guía de aprendizaje: replicar svo-pi F1 y F2 paso a paso

Esta guía explica **qué intenta conseguir el proyecto, por qué está hecho así y en qué orden
replicarlo a mano**. No sustituye a [README_F1_F2.md](README_F1_F2.md), que es la guía de
ejecución con todos los comandos. Aquí está el "para qué" y el plan de estudio.

---

## 1. Qué es el proyecto

### 1.1 El objetivo final

El proyecto quiere implementar **SVO** (*Semi-direct Visual Odometry*, Forster et al.) en
Python, con una Raspberry Pi 4, una cámara y una IMU barata (MPU-6050/9250).

- **Odometría visual (VO):** estimar cómo se mueve la cámara (posición y orientación en cada
  instante) usando sólo las imágenes.
- **Odometría visual-inercial (VIO):** lo mismo, pero ayudándose de la IMU (giróscopo y
  acelerómetro). La IMU aporta la escala métrica y aguanta los movimientos rápidos, donde las
  imágenes salen movidas.

Las fases previstas son estas:

| Fase | Qué es | Estado |
|---|---|---|
| **F1** | Adquisición: grabar imágenes e IMU con timestamps fiables | Código hecho, sin ejecutar en la Pi |
| **F2** | Calibración: ruido de la IMU, óptica de la cámara, posición y retardo cámara↔IMU | Código hecho, sin ejecutar |
| F3 | SVO propio en Python, evaluado sobre el dataset EuRoC | No existe aún |
| F4 | Añadir la IMU (preintegración) | No existe aún |
| F5 | Usarlo con datos propios | No existe aún |

> El README menciona "Ec. 7", "Ec. 13" y "Tabla II" de un paper. Averigua **qué artículo
> exacto** es (seguramente uno de los de SVO de Forster et al.): es el documento que define
> los criterios de F3 en adelante.

### 1.2 Por qué hacen falta F1 y F2

Un algoritmo de VIO fusiona dos sensores. Para fusionarlos tiene que saber tres cosas:

1. **Cuándo** se tomó cada medida, en un mismo reloj y con precisión de microsegundos para
   la IMU y de milisegundos para la cámara. Girando a 1 rad/s, 10 ms de error son 0.57°.
2. **Cómo es cada sensor:** la óptica de la cámara (intrínsecos) y el ruido de la IMU.
3. **Cómo están colocados uno respecto al otro:** la rotación y traslación entre cámara e
   IMU (`T_c_i`) y el retardo temporal entre sus relojes (`t_d`).

Si cualquiera de las tres está mal, el VIO diverge, y no sabrás si el fallo está en tu
algoritmo o en tus datos. F1 resuelve el punto 1 y F2 los puntos 2 y 3. Al final de F2 se
comprueba todo con un SVO de referencia que ya funciona (`rpg_svo_pro_open`): si ese SVO
trackea con tus datos, los datos son buenos.

---

## 2. Conceptos que vas a necesitar

Léelos por encima ahora y vuelve a ellos cuando llegues al paso que los usa.

### Sensores y hardware

- **IMU MEMS (MPU-6050).** Chip con giróscopo (velocidad angular, rad/s) y acelerómetro
  (aceleración específica, m/s²; en reposo mide la gravedad, ≈ 9.81). Se habla con ella por
  **I2C**: se leen y escriben *registros* de 8 bits.
- **Oscilador propio.** La IMU muestrea con su propio reloj, no con el de la Pi. Si le pides
  200 Hz, en realidad da 197 o 203 Hz (±1–2 %). Por eso nunca hay que suponer Δt = 5 ms.
- **FIFO.** Memoria interna de la IMU (1024 B en la 6050) donde se van guardando las muestras.
  Cada muestra son 12 bytes (3 ejes de acelerómetro y 3 de giróscopo, int16). Gracias a ella
  Python puede leer a ráfagas cada 10 ms sin perder muestras.
- **Pin INT (DATA_RDY).** Cada vez que la IMU escribe una muestra, emite un pulso en este pin.
  Conectado a un GPIO de la Pi, el **kernel de Linux** apunta la hora exacta del pulso, sin
  depender de cuándo se despierte Python.
- **Rolling shutter.** Las cámaras normales de la Pi (v1, v2, v3) exponen la imagen fila a
  fila. La última fila se toma unos 20–30 ms después que la primera, y si giras la cámara la
  imagen se deforma (≈ 12 px a 1 rad/s). Las cámaras *global shutter* exponen toda la imagen
  a la vez. Kalibr y SVO suponen global shutter, y de ahí sale el "problema abierto".

### Tiempo

- **Relojes de Linux.** `CLOCK_MONOTONIC` nunca salta hacia atrás; es el que usan libcamera
  (cámara) y gpiod (IMU). `CLOCK_REALTIME` es la hora de pared y puede saltar con NTP.
  `CLOCK_BOOTTIME` es MONOTONIC más el tiempo en suspensión. El proyecto verifica que los dos
  sensores usan el mismo reloj.
- **Jitter.** Variación aleatoria en el instante en que se sella un evento. Python tiene un
  jitter de milisegundos; la IRQ del kernel, de decenas de µs.
- **`t_d` (retardo cámara-IMU).** Convención del proyecto y de Kalibr: `t_imu = t_cam + t_d`.
  Es el número que hay que sumar a los timestamps de la cámara para que cuadren con los de la
  IMU.

### Calibración

- **Intrínsecos.** Focal (fx, fy), centro óptico (cx, cy) y distorsión de la lente. Se usa el
  modelo `pinhole-radtan`.
- **Error de reproyección.** Se proyecta el patrón conocido con la calibración estimada y se
  mide a cuántos píxeles queda de lo detectado. El objetivo del proyecto es menos de 0.3 px.
- **Extrínseca `T_c_i`.** Matriz 4×4 (rotación más traslación) que pasa un punto del sistema
  de la IMU al de la cámara: `p_c = T_c_i · p_i`. Cuidado: cada herramienta usa su propia
  convención y a veces es la inversa.
- **Allan variance.** Técnica para caracterizar el ruido de la IMU a partir de una grabación
  estática larga (3 h). Da dos números por sensor: el ruido blanco (σ_n) y la deriva lenta
  del sesgo (σ_b, *random walk*). Kalibr y los filtros VIO los necesitan.
- **AprilGrid.** Patrón impreso de marcadores que Kalibr detecta en las imágenes.
- **Kalibr.** Herramienta de ETH Zúrich para calibrar cámaras y cámara-IMU. Funciona en ROS 1,
  así que se usa dentro de Docker. Lee los datos en **rosbag**, el formato de grabación de
  ROS.

### Formato de datos

- **EuRoC / ASL.** Formato estándar de datasets VIO, que es el que produce F1:

  ```
  mav0/
    cam0/data.csv        #timestamp [ns],filename
    cam0/data/<t>.png
    imu0/data.csv        #timestamp [ns], wx, wy, wz [rad/s], ax, ay, az [m/s²]
  ```

  Grabar en el mismo formato que el dataset público EuRoC permite usar las mismas
  herramientas y el mismo código de carga con datos propios y con datos de referencia.

---

## 3. Cómo fluyen los datos

```
 ┌─────────────── Raspberry Pi (F1) ────────────────┐      ┌──────────────── PC (F2) ─────────────────┐
 │                                                   │      │                                           │
 │  cámara ──Picamera2──┐                            │      │  ruido_imu ──► imu.yaml (σ_n, σ_b)         │
 │                      ├─► grabar.py ──► raw/       │ scp  │                                           │
 │  IMU ──I2C(FIFO)─────┤     (crudo)      │         │ ───► │  preparar_kalibr ─► Docker: crear_bag      │
 │  IMU ──INT(GPIO17)───┘                  ▼         │      │        │                                  │
 │                       construir_euroc ─► mav0/    │      │        ├─► Kalibr intrínsecos              │
 │                       (emparejamiento)   │        │      │        └─► Kalibr cámara-IMU (×3)          │
 │                                          ▼        │      │                 │                         │
 │                       verificar_secuencia         │      │        revisar_kalibr ─► calibracion_final │
 │                       ─► informe_F1.png / .json   │      │                 │                         │
 │                          (+ t_d previo)           │      │        kalibr_a_svo ─► Docker: svo_pro     │
 └───────────────────────────────────────────────────┘      │                 └─► evaluar_baseline       │
                                                            └───────────────────────────────────────────┘
```

Una decisión de diseño importante: **`grabar.py` guarda datos crudos sin interpretar** y todo
el procesado se hace después, offline. Así, si mejoras el algoritmo de timestamps, no hace
falta volver a grabar.

---

## 4. El problema central: los timestamps de la IMU

La parte más difícil de F1 es esta, y conviene entenderla bien antes de programar nada.

### 4.1 El planteamiento

1. La IMU escribe la muestra *i* en su FIFO en el instante real τ_i (que no conocemos).
2. A la vez emite un pulso en INT. El kernel lo sella con su hora `t_s` y le da un número de
   secuencia `s` (1, 2, 3…).
3. Cada 10 ms Python lee de la FIFO las muestras que haya (unas 2) y apunta la hora antes y
   después de leer (`t_antes`, `t_despues`).

Al final tenemos una lista de **muestras** sin hora y una lista de **flancos** con hora. Hay
que emparejarlas.

### 4.2 Por qué no basta con emparejarlas en orden

Al arrancar puede sobrar un flanco (el pulso llegó antes de activar la FIFO) o una muestra.
Entonces la muestra *i* corresponde al flanco *i + m*, con *m* un entero desconocido
(normalmente 0 o ±1). Si eliges mal *m*, todos los timestamps quedan desplazados 5 ms.

### 4.3 Cómo se encuentra m

Cada lectura de la FIFO da una pista. Si en el instante *t* la FIFO tenía *C* muestras en
total (acumuladas):

- **R1:** la muestra *C − 1* ya existía, así que su flanco tiene que ser anterior a `t_despues`.
- **R2:** la muestra *C* todavía no existía, así que su flanco tiene que ser posterior a
  `t_antes`.

Con el *m* correcto casi ninguna lectura viola R1 o R2. Con un *m* erróneo, alrededor del 90 %
las viola. Se prueban varios *m* y se queda el que menos violaciones tiene. Con miles de
lecturas no hay duda.

### 4.4 Suavizado

Los flancos tienen un jitter de decenas de µs (la IRQ no siempre tarda lo mismo). Como el
oscilador de la IMU es muy estable a corto plazo, los timestamps reales siguen una recta
`t = a + b·i`. Ajustando esa recta por bloques de 10 s:

- el jitter baja a menos de 1 µs;
- `b` da el **periodo real** de la IMU (p. ej. 4.95 ms en lugar de 5);
- si el kernel perdió algún flanco, la recta rellena el hueco.

El retardo medio de la IRQ (~10 µs) no se elimina, pero es constante y acaba sumado a `t_d`.

### 4.5 Qué t_d esperar

`t_d` no es cero porque cada sensor "llega tarde" de forma distinta:

- La IMU filtra la señal (DLPF) y eso la retrasa unos 4.8 ms.
- La cámara sella el *inicio de la lectura*, no el centro de la exposición, y con rolling
  shutter el centro de la imagen se lee unos 11.5 ms después.

Con los valores por defecto: `t_d ≈ 4.8 − 2 + 11.5 ≈ +14 ms`. Este valor se comprueba dos
veces de forma independiente: en F1 (correlando el giróscopo con el flujo óptico) y en F2
(Kalibr). Si no coinciden con un margen de pocos ms, hay un error de relojes o de
emparejamiento.

---

## 5. Mapa de ficheros

Ordenados por el orden recomendado de estudio. Las líneas indican el tamaño aproximado.

| Orden | Fichero | Qué hace | Líneas |
|---|---|---|---|
| 1 | [config/captura.yaml](config/captura.yaml) | Todos los parámetros de F1. Léelo entero: está muy comentado | 68 |
| 2 | [captura/comun.py](captura/comun.py) | Utilidades: cargar config, leer relojes, CSV | 62 |
| 3 | [captura/mpu6050.py](captura/mpu6050.py) | Driver de la IMU: registros, FIFO, decodificación | 243 |
| 4 | [captura/emparejamiento_imu.py](captura/emparejamiento_imu.py) | Algoritmo de la §4 (puro, sin hardware) | 185 |
| 5 | [captura/identificar_hw.py](captura/identificar_hw.py) | Detecta cámara, IMU y prueba el pin INT | 137 |
| 6 | [captura/grabar.py](captura/grabar.py) | Grabador con 3 hilos, escribe `raw/` | 429 |
| 7 | [captura/construir_euroc.py](captura/construir_euroc.py) | `raw/` → `mav0/` usando el emparejamiento | 249 |
| 8 | [captura/verificar_secuencia.py](captura/verificar_secuencia.py) | Criterios de salida F1, histogramas, t_d previo | 449 |
| 9 | [config/calibracion.yaml](config/calibracion.yaml) | Parámetros de F2 | 58 |
| 10 | [calibracion/allan.py](calibracion/allan.py) | Cálculo de la desviación de Allan | 77 |
| 11 | [calibracion/ruido_imu.py](calibracion/ruido_imu.py) | Allan o datasheet → `imu.yaml` para Kalibr | 166 |
| 12 | [calibracion/kalibr_io.py](calibracion/kalibr_io.py) | Lee y escribe los ficheros de Kalibr | 148 |
| 13 | [calibracion/preparar_kalibr.py](calibracion/preparar_kalibr.py) | Prepara la carpeta que monta Docker | 120 |
| 14 | [docker/kalibr/](docker/kalibr/) | Scripts que se ejecutan dentro del contenedor | ~50 |
| 15 | [calibracion/revisar_kalibr.py](calibracion/revisar_kalibr.py) | Criterios F2 y `calibracion_final.yaml` | 153 |
| 16 | [calibracion/kalibr_a_svo.py](calibracion/kalibr_a_svo.py) | Convierte la calibración al formato de svo_pro | 112 |
| 17 | [docker/svo_pro/](docker/svo_pro/) | Imagen Docker y lanzador de svo_pro | ~150 |
| 18 | [calibracion/evaluar_baseline.py](calibracion/evaluar_baseline.py) | ¿svo_pro trackea con mis datos? | 121 |
| — | [tests/](tests/) | 14 tests sintéticos. Son la mejor documentación de qué debe hacer cada pieza | ~540 |

### Qué está probado y qué no

- **Probado** (con datos sintéticos): driver con bus falso, emparejamiento, construcción y
  verificación de una sesión simulada, Allan y lectura y conversión de ficheros de Kalibr.
- **Sin probar nunca:** el grabador con hardware real y la construcción de las dos imágenes
  Docker.

Al replicar también estás validando el código. Si tu versión y la generada no coinciden,
**la equivocada puede ser cualquiera de las dos**.

---

## 6. Hoja de ruta

### Método para cada paso

1. **Aprende:** lee el concepto (sección 2 y 4, datasheet, documentación).
2. **Programa** tu versión sin mirar el código del proyecto.
3. **Prueba** con el criterio de "hecho" del paso.
4. **Compara** con el fichero de referencia: ¿qué hizo distinto? ¿por qué?

Trabaja en una carpeta aparte (p. ej. `mi-svo-pi/`) con `git init` desde el primer día y un
commit por paso.

Los pasos del bloque A no necesitan la Pi. Puedes empezar hoy e intercalarlos con los demás.

---

### Bloque A: en el PC, sin hardware

#### A0. Conocer el formato de destino

- **Objetivo:** saber exactamente qué tiene que producir F1.
- **Haz:** descarga la secuencia `MH_01_easy` de EuRoC MAV. Abre `cam0/data.csv`,
  `imu0/data.csv` y los `sensor.yaml`.
- **Programa:** un script que cargue la secuencia y dibuje:
  - el giróscopo y el acelerómetro frente al tiempo;
  - el histograma de Δt de la IMU y de la cámara;
  - una imagen cualquiera.
- **Hecho cuando:** sabes decir la frecuencia real de cada sensor y qué significa cada columna.
- **Por qué:** EuRoC es tu "dataset que sabes que funciona". Lo usarás en A4 y en F3.

#### A1. Teoría de tiempos y convenciones

- **Haz:** lee las secciones 2 y 4 de esta guía y los §1.4 y §2.6 del README.
- **Ejercicio en papel:**
  - si `p_c = T_c_i · p_i`, ¿cómo se escribe `T_i_c`?
  - si `t_imu = t_cam + t_d` y otra herramienta usa `t_imu = t_cam − delay`, ¿qué vale
    `delay`?
- **Hecho cuando:** puedes explicar en voz alta por qué el proyecto necesita el pin INT.

#### A2. Emparejamiento flanco↔muestra con datos sintéticos

Es el paso algorítmico más importante de F1.

- **Programa primero un simulador** (así sabes la respuesta correcta):

  ```python
  # Esqueleto orientativo: complétalo tú
  periodo_real = 4.95e6            # ns (la IMU no da 5 ms exactos)
  n = 12000                        # 60 s a ~200 Hz
  tau = t0 + periodo_real * np.arange(n)        # instante real de cada muestra
  lat = rng.exponential(30e3, n)                # latencia de la IRQ, siempre ≥ 0
  m_real = 1                                    # sobra un flanco al arrancar
  # flancos: seq = i + m_real, t = tau + lat
  # lecturas cada ~10 ms (con jitter): t_antes, t_despues, cuántas muestras había
  ```

- **Programa luego:**
  1. `puntuar(m)`: cuenta violaciones de R1 y R2 (§4.3);
  2. búsqueda del mejor *m* en un rango ±20;
  3. ajuste lineal por bloques de 10 s con rechazo de outliers (§4.4).
- **Hecho cuando:**
  - recuperas `m_real`;
  - el error entre timestamp estimado y `tau` es < 5 µs;
  - el periodo estimado ≈ 4.95 ms;
  - si borras un 1 % de flancos, la recta los rellena.
- **Compara con:** [captura/emparejamiento_imu.py](captura/emparejamiento_imu.py) y
  [tests/test_emparejamiento_imu.py](tests/test_emparejamiento_imu.py).
- **Extra:** el "plan B" sin pin INT (`estimar_sin_interrupcion`). ¿Qué precisión consigues
  sólo con `t_antes` y `t_despues`?

#### A3. Allan variance

- **Aprende:** qué es la desviación de Allan y cómo se lee la gráfica log-log: la pendiente
  −1/2 corresponde al ruido blanco y la +1/2 al random walk.
- **Programa:**
  1. genera 3 h de señal sintética a 200 Hz: ruido blanco con σ_n conocido más un sesgo
     que deriva con σ_b conocido;
  2. calcula σ(τ) para τ de 0.01 s a 1000 s;
  3. extrae σ_n = σ(τ = 1 s) en la recta −1/2 y σ_b en la recta +1/2.
- **Hecho cuando:** recuperas los parámetros con un error menor del 10–20 %.
- **Compara con:** [calibracion/allan.py](calibracion/allan.py) y
  [tests/test_allan.py](tests/test_allan.py).
- **Después:** la "vía rápida" del datasheet (`modo_datasheet` en
  [calibracion/ruido_imu.py](calibracion/ruido_imu.py)): convertir °/s/√Hz a rad/s/√Hz y
  µg/√Hz a m/s²/√Hz.

#### A4. Kalibr y svo_pro con datos de EuRoC

- **Objetivo:** aprender las herramientas con datos que sabes que funcionan.
- **Haz:**
  1. instala Docker Desktop (WSL2) y construye la imagen de Kalibr (README §2.1, ojo con
     `core.autocrlf=false`);
  2. descarga las secuencias de **calibración** de EuRoC (`cam_april`, `imu_april`) y ejecuta
     los intrínsecos y la calibración cámara-IMU;
  3. compara tu resultado con el `sensor.yaml` oficial de EuRoC;
  4. construye la imagen de svo_pro ([docker/svo_pro/Dockerfile](docker/svo_pro/Dockerfile))
     y ejecútala sobre `MH_01`.
- **Hecho cuando:** Kalibr da una reproyección < 0.3 px sobre EuRoC y svo_pro trackea `MH_01`.
- **Por qué:** cuando más adelante falle con tus datos, sabrás que la herramienta funciona y
  que el problema está en los datos.

---

### Bloque B: en la Raspberry Pi

#### B1. Hardware y primer contacto

- **Haz:**
  1. prepara la Pi (README §1.1: I2C a 400 kHz, venv con `--system-site-packages`);
  2. cablea la IMU **al conector de 40 pines**, no al de cámara (tabla en README §0.1);
  3. ejecuta `i2cdetect -y 1` (debe aparecer `68`) y `rpicam-hello --list-cameras`.
- **Programa:** con `smbus2`:
  - lee `WHO_AM_I` (registro 0x75; esperado 0x68 en la 6050, 0x71 en la 9250);
  - despierta la IMU (`PWR_MGMT_1 = 0x01`);
  - lee los 14 bytes desde 0x3B en un bucle.
- **Hecho cuando:**
  - en reposo |a| ≈ 9.81 m/s²;
  - al girar sobre cada eje cambia el eje correcto del giróscopo;
  - sabes el signo de cada eje y lo tienes dibujado sobre la placa.

#### B2. Driver de la IMU

- **Aprende:** el *MPU-6000/6050 Register Map*, en particular `SMPLRT_DIV`, `CONFIG` (DLPF),
  `GYRO_CONFIG`, `ACCEL_CONFIG`, `FIFO_EN`, `USER_CTRL` e `INT_ENABLE`.
- **Programa:** una clase con `configurar()` (200 Hz, DLPF 3, ±1000 °/s, ±8 g),
  `arrancar()`, `leer_conteo_fifo()`, `leer_tramas(n)` y una función que decodifique bytes →
  unidades físicas.
- **Prueba sin hardware:** escribe un **bus falso** (un diccionario de registros que imita a
  `smbus2`). Comprueba qué registros escribe `configurar()` y que la decodificación de una
  trama conocida da el valor esperado.
- **Hecho cuando:** pasan tus tests y con la IMU real lees muestras de la FIFO.
- **Compara con:** [captura/mpu6050.py](captura/mpu6050.py) y
  [tests/test_mpu6050.py](tests/test_mpu6050.py).

#### B3. Experimento: ¿por qué hace falta la interrupción?

Este experimento justifica todo el diseño de F1. No te lo saltes.

- **Programa (versión 1):** lee la FIFO cada 10 ms y asigna a cada muestra
  `time.monotonic_ns()` en Python, repartiendo las muestras de cada lectura.
- **Programa (versión 2):** con **gpiod v2** (`pip install gpiod`, no el de apt), escucha
  flancos de subida en GPIO17 con `event_clock = MONOTONIC` y guarda `line_seqno` y
  `timestamp_ns`.
- **Dibuja:** el histograma de Δt de ambas versiones.
- **Hecho cuando:** ves que la versión 2 tiene un jitter mucho menor y mides la frecuencia
  real de tu IMU (no será 200.0 Hz).
- **Compara con:** `probar_interrupcion` en [captura/identificar_hw.py](captura/identificar_hw.py).

#### B4. Cámara con Picamera2

- **Aprende:** los controles de Picamera2 (`ExposureTime`, `AnalogueGain`, `AwbEnable`,
  `ColourGains`, `FrameDurationLimits`) y los metadatos (`SensorTimestamp`).
- **Programa:**
  1. identifica tu sensor y sus modos (así sabrás qué cámara tienes);
  2. captura a 20 fps con exposición, ganancia y balance de blancos **fijos**;
  3. guarda el canal Y en PNG y un CSV con `SensorTimestamp`, la hora de recepción y la
     exposición y ganancia que reportan los metadatos.
- **Hecho cuando:** 60 s sin huecos en Δt, exposición y ganancia constantes en todos los
  frames y `t_recepcion − SensorTimestamp` positivo y menor de 150 ms (prueba de que ambos
  están en MONOTONIC).
- **Compara con:** `identificar_camara` en [captura/identificar_hw.py](captura/identificar_hw.py)
  y `configurar_camara`, `construir_controles` y `bucle_camara` en
  [captura/grabar.py](captura/grabar.py).

#### B5. Grabador completo

- **Aprende:** hilos en Python (`threading`, `queue`) y por qué aquí funcionan pese al GIL
  (las esperas de libcamera y `cv2.imwrite` liberan el GIL).
- **Programa:**
  - hilo IMU: vacía la FIFO cada 10 ms y drena los flancos del kernel;
  - hilo escritor: guarda los PNG desde una cola;
  - hilo de relojes: muestrea MONOTONIC, BOOTTIME y REALTIME cada segundo;
  - hilo principal: la cámara.
  - Formato de salida: los CSV crudos descritos al principio de `grabar.py`.
  - Gestiona el desbordamiento de la FIFO: por encima del 90 % se reinicia y se abre un
    "segmento" nuevo.
- **Hecho cuando:** grabas 60 s con un solo segmento y sin frames perdidos.
- **Compara con:** [captura/grabar.py](captura/grabar.py), el fichero más largo del proyecto.

#### B6. Crudo → EuRoC → verificación

- **Programa:**
  1. `construir_euroc`: lee `raw/`, aplica tu emparejamiento de A2 y escribe `mav0/` con las
     cabeceras de EuRoC. Los timestamps de cámara se guardan sin corregir.
  2. `verificar`: implementa los criterios del README §1.5 uno a uno, cada uno con su umbral
     en el YAML.
  3. **t_d previo:** calcula la magnitud del flujo óptico entre frames (Lucas-Kanade de
     OpenCV) y la correlación con |ω| para distintos retardos. El máximo es tu `t_d` previo.
- **Prueba sin hardware:** genera una sesión `raw/` sintética con un retardo de cámara
  conocido y comprueba que lo recuperas.
- **Hecho cuando:**
  - con la sesión sintética, recuperas el retardo;
  - con una sesión real "estática" pasan los criterios;
  - con una sesión real de "sacudidas", el t_d previo sale cerca de +14 ms.
- **Compara con:** [captura/construir_euroc.py](captura/construir_euroc.py),
  [captura/verificar_secuencia.py](captura/verificar_secuencia.py) y
  [tests/test_construir_y_verificar.py](tests/test_construir_y_verificar.py).

**Fin de F1.**

---

### Bloque C: calibración con tus datos (F2)

#### C1. Ruido de la IMU

- **Haz:** empieza con la vía datasheet ×10 para no bloquearte. Cuando el grabador sea
  estable, graba 3 h estáticas (`--solo-imu`) y aplica tu Allan de A3.
- **Hecho cuando:** tienes un `imu.yaml` para Kalibr. Más adelante, en C3, valida que los
  residuos normalizados de Kalibr tienen σ ≈ 1.

#### C2. Intrínsecos

- **Haz:** imprime el AprilGrid en A3 al 100 %, pégalo en algo rígido y **mide** el lado del
  tag con regla. Graba 90 s moviendo **muy despacio** (rolling shutter) y cubriendo todo el
  campo, incluidas las esquinas.
- **Programa:** tu `preparar_kalibr`: recorta la sesión y escribe `aprilgrid.yaml`,
  `imu.yaml` y los scripts con finales de línea LF (no CRLF).
- **Hecho cuando:** reproyección RMS < 0.3 px.

#### C3. Cámara-IMU (al menos 3 veces)

- **Haz:** 3 grabaciones independientes (vuelve a coger el montaje entre ellas). Al principio
  y al final, 3 s quieto; en medio, excita los 3 ejes de giro y los 3 de traslación.
- **Programa:** tu `revisar_kalibr`:
  - lee los resultados de Kalibr;
  - calcula la mediana y la dispersión de `t_d`;
  - calcula el ángulo y la distancia entre las `T_c_i` de cada calibración;
  - escribe `calibracion_final.yaml`.
- **Ejercicio clave:** deduce en papel por qué el `T_B_C` de svo_pro es `(T_c_i)⁻¹` y por qué
  su `delay_imu_cam` es `−t_d`. Luego compáralo con
  [calibracion/kalibr_a_svo.py](calibracion/kalibr_a_svo.py).
- **Hecho cuando:**
  - σ(t_d) ≤ 1 ms;
  - el `t_d` de Kalibr coincide con el t_d previo de B6 con un margen de pocos ms.
  - La reproyección no bajará de 0.3 px con rolling shutter: anótalo como problema abierto.

#### C4. Baseline con svo_pro

- **Haz:** graba 60 s "de uso" (habitación con textura, **trasladando** la cámara al
  principio para que inicialice). Ejecuta svo_pro primero en modo VO y después en VIO, igual
  que en A4.
- **Diagnóstico:**
  - si VO falla, el problema es de imagen (intrínsecos, blur, exposición, rolling shutter);
  - si VO funciona y VIO no, el problema está en `T_c_i`, `t_d` o el ruido de la IMU.
- **Hecho cuando:** pasa `evaluar_baseline`: ≥ 90 % de frames en tracking y mediana de matches
  ≥ 50.

**Fin de F2.**

---

## 7. Orden recomendado

```
A0 → A1 → A2 ──────────────────┐
           (tienes la Pi)      │
           B1 → B2 → B3 → B4 → B5 → B6 ──► C1 → C2 → C3 → C4
                               │
           A3, A4 ─────────────┘ (cuando no tengas la Pi a mano)
```

## 8. Checklist

- [ ] A0 EuRoC cargado y entendido
- [ ] A1 Convenciones de tiempo y SE(3) claras
- [ ] A2 Emparejamiento sintético: m correcto, error < 5 µs
- [ ] A3 Allan sintético: parámetros recuperados
- [ ] A4 Kalibr y svo_pro funcionando con EuRoC
- [ ] B1 IMU cableada, WHO_AM_I correcto, ejes identificados
- [ ] B2 Driver con tests de bus falso
- [ ] B3 Histogramas polling frente a INT; frecuencia real medida
- [ ] B4 Cámara identificada; 60 s sin huecos con controles fijos
- [ ] B5 Grabador: 60 s, 1 segmento, 0 frames perdidos
- [ ] B6 EuRoC propio verificado; t_d previo ≈ +14 ms
- [ ] C1 `imu.yaml` (datasheet, y luego Allan)
- [ ] C2 Intrínsecos < 0.3 px
- [ ] C3 3 calibraciones cámara-IMU; t_d repetible y coherente con B6
- [ ] C4 svo_pro trackea en VO y VIO con tus datos

## 9. Recursos

- EuRoC MAV Dataset (secuencias y datos de calibración): proyecto ASL, ETH Zúrich.
- *MPU-6000/MPU-6050 Register Map and Descriptions*, Rev 4.2, y *Product Specification*
  (InvenSense/TDK).
- Documentación de Picamera2 (Raspberry Pi), sobre todo controles y metadatos.
- Documentación de libgpiod v2 y de sus bindings de Python.
- Wiki de Kalibr (`ethz-asl/kalibr`): *Camera-IMU calibration*, *IMU Noise Model*.
- `rpg_svo_pro_open` (Universidad de Zúrich) y los papers de SVO de Forster et al.
- IEEE Std 952 (Allan variance para giróscopos), para la teoría del ruido.
