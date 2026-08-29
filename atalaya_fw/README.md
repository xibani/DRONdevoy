# ATALAYA — framework VIO v0.1

Refactor a paquete del pipeline construido en las Fases 1–7 del curso, extendido
para trabajar con **tus datos** (ArduPilot + vídeo) además de EuRoC. Mantiene los
convenios del curso al milímetro (`anexo_convenios.md`): mundo z-arriba con
`G_W = (0,0,-9.81)`, Hamilton `(x,y,z,w)` interno, `T_A_B` lleva puntos de B a A,
perturbación derecha `R = R̄·Exp(δθ)`, estado de error `[δp, δv, δθ, δb_g, δb_a]`.

Incorpora los hallazgos validados de las fases (documentados en las cabeceras de
cada módulo): las tres correcciones al guion del ESKF (Fase 5), el modo
**dirección + gating χ²** como defecto (el estimador de escala absorbe outliers y
neutraliza el χ²), el paralaje **derotado** como puerta de keyframe, tracking
sobre imagen cruda con indistorsión solo de keypoints (Fase 2), y `b_a = 0` en la
inicialización (el tramo estático es degenerado para `b_a`, Fase 3.9).

## Instalación

```bash
pip install numpy scipy opencv-python pandas pyyaml matplotlib
pip install pymavlink            # solo para logs de ArduPilot
```

No hay setup.py: se ejecuta desde la carpeta que contiene `atalaya/`
(`python -m atalaya ...`) o añadiendo esa carpeta al `PYTHONPATH`.

## Primer comando, siempre

```bash
python -m atalaya selftest
```

Valida TODO en sintético sin datos: jacobianos por diferencias finitas
(incluido el test negativo del H erróneo del guion), ESKF sobre trayectoria
analítica (d² mediana ≈ 5.35), robustez del gating con 5 % de medidas
corruptas, front-end sobre imágenes renderizadas e inicialización estática.
Si esto pasa y tus datos fallan, el problema está en los datos o la config
(unidades, extrínseca, offset temporal), no en la matemática.

## Flujo con un vuelo propio (ArduPilot)

1. **Inspeccionar el log** — qué mensajes trae y a qué tasa, para decidir
   `dataset.imu.mensaje` (IMU vs GYR+ACC), `instancia` y `gt.fuente`:
   ```bash
   python -m atalaya inspeccionar configs/ardupilot_ejemplo.yaml
   ```
2. **Componer los frames del vídeo** (se escriben una vez, formato EuRoC):
   ```bash
   python -m atalaya extraer-frames configs/ardupilot_ejemplo.yaml
   ```
3. **Estimar el offset temporal cámara↔IMU** (decenas de ms en un setup
   Pi + autopiloto; correlación 3D por ejes del giro visual vs gyro):
   ```bash
   python -m atalaya offset configs/ardupilot_ejemplo.yaml --t-ini 20 --t-fin 60
   ```
   Copia el resultado en `dataset.camara.offset_temporal_s`. Usa un tramo con
   giros claros: sin dinámica angular el lag es inobservable y se avisa.
4. **Ejecutar** (front-end → ESKF → dead reckoning → informe):
   ```bash
   python -m atalaya ejecutar configs/ardupilot_ejemplo.yaml
   ```

El informe (`salida.directorio`) trae `trayectoria.png`, `filtro.png`,
`frontend.png`, `est.tum`/`gt.tum` (para `evo`), `resumen.txt` y
`resultado.npz` con el mismo contrato de claves que los `.npz` de las fases
(`t, p_est, q_est_wxyz, bg, ba, ...`).

## Qué vigilar en el informe

* **tasa de aceptación** > 80 % y **d² mediana** ≈ 5.35 (χ²(6)): >>5 hay bug de
  jacobiano/extrínseca/unidades; <<1, `Rm` pesimista.
* Con GT de ArduPilot, el ATE mide **acuerdo con el EKF3**, no error absoluto.
* Con inicialización estática, yaw y origen son arbitrarios: solo las métricas
  alineadas (Umeyama) tienen sentido, y el informe lo indica.
* Si "ATE razonable pero todo tiembla": offset temporal (paso 3).
* Si diverge al añadir la cámara: `T_body_cam` invertida (chuleta A.7).

## Estructura

```
atalaya/
  geometria.py       SO(3)/SE(3), Umeyama, ATE, TUM (≡ anexo_utils)
  sensores.py        CamaraPinhole (radtan/fisheye), ParamsImu
  datasets/
    base.py          contrato Secuencia (imu_entre exacto, tramo estático, GT)
    euroc.py         formato ASL
    ardupilot.py     DataFlash vía pymavlink (IMU|GYR+ACC, ATT+POS, CAM/TRIG)
    video.py         vídeo -> frames + data.csv (fps | csv | triggers del log)
    generico.py      CSVs con mapeo de columnas y unidades
  frontend.py        KLT con IDs, FB-check, paralaje derotado, pose relativa
  eskf.py            ESKF 15+6 con Joseph, gating χ², modos direccion/escala
  inicializacion.py  estática (b_g + gravedad, b_a=0) o desde GT
  time_offset.py     offset cámara↔IMU por correlación 3D paso-alto
  evaluacion.py      informe: PNGs, TUM, resumen, resultado.npz
  selftest.py        validación sintética completa
  cli.py             selftest | inspeccionar | extraer-frames | offset | ejecutar
configs/             euroc_mh01.yaml, ardupilot_ejemplo.yaml, generico_ejemplo.yaml
```

## Validación de esta versión (reproducible)

* `selftest`: jacobianos a <1e-10 de las diferencias finitas; el H del guion se
  detecta con error 2.0 (test negativo); d² mediana 5.28; con 5 % de corruptas,
  dirección+gating 0.93 m vs 1.79 m sin gating; front-end sintético con error de
  rotación 0.075° y de dirección 0.99°; init estática 0.003° de roll/pitch.
* End-to-end sobre un dataset sintético "estilo datos propios" (vídeo AVI +
  IMU CSV en µs y deg/s + GT), 24 s: 96.8 % de aceptación, ATE alineado
  0.08–0.13 m, actitud 1.6°, mejora 56x sobre IMU sola.
* Estimador de offset: recupera desfases inyectados de 0/-40/+75 ms con error
  ±0.2 ms (correlación 1.000). Nota honesta: con rotación de ancho de banda
  ~0.1 Hz el lag es inobservable por cualquier método; hace falta dinámica
  angular (los vuelos reales la traen).

## Límites conocidos de v0.1 y siguiente paso

Es el ESKF loosely-coupled de la Fase 5: una medida relativa no acota la
posición absoluta (deriva tipo random walk, esperada y medida). El back-end
tightly-coupled (MSCKF de la 6A / grafo iSAM2 de la 6B) es el siguiente módulo
a portar; la interfaz ya está preparada: `construir_medidas` produce las
mismas medidas y el `resultado.npz` mantiene el contrato entre fases.

## Interfaz gráfica (opcional)

`servidor_api.py` expone toda la lógica como API REST local para una interfaz
web (el prompt para generarla con Lovable/Stitch está en
`prompt_interfaz_atalaya.md`):

```bash
pip install fastapi "uvicorn[standard]"
python servidor_api.py           # http://localhost:8420
```

Endpoints: `/api/estado`, `/api/configs` (leer/guardar/editar campo),
`/api/trabajos` (lanza selftest/inspeccionar/extraer-frames/offset/ejecutar
como subprocesos con log en vivo), `/api/resultados` (sirve PNGs/TUM/npz) y
`/api/explorar` (selector de archivos). Solo para localhost: no lo expongas a
Internet.
