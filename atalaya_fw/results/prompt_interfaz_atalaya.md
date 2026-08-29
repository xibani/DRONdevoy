# PROMPT PARA LOVABLE / GOOGLE STITCH — Interfaz de ATALAYA

> Copia desde aquí hasta el final en el generador. No necesita nada más:
> incluye el contrato completo de la API contra la que debe funcionar.

---

Construye una aplicación web (SPA, sin backend propio, sin base de datos, sin
autenticación) llamada **ATALAYA** que es el panel de control de un sistema de
odometría visual-inercial (VIO) para drones. Toda la lógica pesada ya existe en
un servidor Python local; la app SOLO consume su API REST. La app debe estar
íntegramente en **español**.

## Contexto (para que los textos de la interfaz tengan sentido)

El usuario procesa vuelos de dron: un log del autopiloto (ArduPilot, archivo
`.BIN`) con datos de IMU, y un vídeo de la cámara a bordo. El sistema fusiona
ambos para estimar la trayectoria 3D del dron. El flujo de trabajo real tiene
5 pasos y la interfaz debe guiarlo:

1. **Selftest** — validar la instalación (tests matemáticos sintéticos).
2. **Inspeccionar** — ver qué mensajes trae el log (qué IMUs, a qué Hz, si hay
   GPS/actitud/triggers de cámara), para decidir la configuración.
3. **Extraer frames** — convertir el vídeo en imágenes con timestamps.
4. **Offset** — estimar el desfase temporal entre cámara e IMU (milisegundos)
   y aplicarlo a la configuración.
5. **Ejecutar** — correr el pipeline completo y ver el informe (gráficas de
   trayectoria, métricas de error, archivos descargables).

Cada paso es un "trabajo" que el servidor ejecuta en segundo plano; puede
tardar de segundos a varios minutos y emite un log de texto en vivo.

## Configuración de conexión

- URL base de la API: `http://localhost:8420`, pero debe ser **editable** por
  el usuario (campo en un panel de ajustes, persistido en localStorage).
- Al arrancar, la app llama a `GET /api/estado`. Si falla, mostrar una pantalla
  de "Servidor no disponible" con las instrucciones exactas:
  `cd atalaya_fw && python servidor_api.py` y un botón "Reintentar".
- CORS ya está habilitado en el servidor; usa `fetch` normal.

## Contrato de la API (exacto; no inventes endpoints ni campos)

### Estado
`GET /api/estado` →
```json
{"ok": true, "version": "0.1.0", "raiz": "/ruta/atalaya_fw",
 "pymavlink": true, "configs": 3}
```
Si `pymavlink` es `false`, mostrar un aviso persistente: "pymavlink no está
instalado: no se podrán leer logs de ArduPilot (`pip install pymavlink`)".

### Configuraciones (archivos YAML)
- `GET /api/configs` → `["ardupilot_ejemplo.yaml", "euroc_mh01.yaml", ...]`
- `GET /api/configs/{nombre}` →
  `{"nombre": "...", "texto": "<yaml crudo>", "contenido": {<yaml parseado a JSON>}}`
- `PUT /api/configs/{nombre}` con body `{"texto": "<yaml>"}` — guarda; si el
  YAML es inválido responde **422** con `{"detail": "YAML inválido: ..."}`;
  mostrar ese detalle bajo el editor.
- `POST /api/configs/{nombre}/campo` con body
  `{"ruta": "dataset.camara.offset_temporal_s", "valor": -0.137}` — cambia un
  solo campo. **OJO**: reescribe el YAML y pierde los comentarios; úsalo SOLO
  para el botón "Aplicar offset" del paso 4, avisando de ello. Para todo lo
  demás, editar el texto completo con PUT.

### Trabajos (ejecución en segundo plano)
- `POST /api/trabajos` con body:
```json
{"comando": "selftest" | "inspeccionar" | "extraer-frames" | "offset" | "ejecutar",
 "config": "ardupilot_ejemplo.yaml",
 "t_ini": 20.0, "t_fin": 60.0, "rango": 2.0,
 "sin_dead_reckoning": false}
```
  - `config` es obligatorio para todos menos `selftest`.
  - `t_ini`, `t_fin`, `rango` solo aplican a `offset` (opcionales).
  - `sin_dead_reckoning` solo aplica a `ejecutar`.
  - Respuesta: `{"id": "abc123..."}`.
- `GET /api/trabajos` → lista de trabajos SIN el log:
  `[{"id","comando","config","estado","resultado","inicio","fin","cmd","codigo"?}]`
  con `estado` ∈ `en_cola | corriendo | ok | error` y `inicio`/`fin` como
  epoch en segundos (float) o null.
- `GET /api/trabajos/{id}?desde=N` → lo mismo más:
  - `log`: SOLO los caracteres a partir de la posición `N` (sondeo
    incremental: la app acumula y en la siguiente llamada pasa
    `desde=log_total`).
  - `log_total`: longitud total actual del log.

  **Sondear cada 1 s mientras `estado` sea `en_cola` o `corriendo`**; parar al
  llegar a `ok`/`error`. El log se muestra en un panel tipo terminal
  (monoespaciada, autoscroll con botón para fijarlo, fondo oscuro).

- Campo `resultado` (objeto, puede venir vacío) según comando:
  - `selftest`: `{"ok": true|false}`.
  - `offset`: `{"offset_s": -0.137, "aviso_contraste": false}`. Si
    `aviso_contraste` es true, avisar: "poca rotación en el tramo elegido; el
    offset no es fiable, elige otro intervalo con giros".
  - `ejecutar`: `{"directorio": "...", "ate_se3": 0.09, "ate_sim3": 0.08,
    "aceptadas_pct": 96.8}` (cualquiera puede faltar).

### Resultados (informes generados)
- `GET /api/resultados` →
```json
[{"nombre": "mi_vuelo", "archivos": ["trayectoria.png", "filtro.png",
  "frontend.png", "resumen.txt", "est.tum", "gt.tum", "resultado.npz"],
  "resumen": "<contenido de resumen.txt o null>", "modificado": 1756480000.0}]
```
- `GET /api/resultados/{nombre}/archivo/{fname}` → el archivo binario. Las
  imágenes se muestran con `<img src=...>` directo a esa URL; el resto con
  botón de descarga.

### Explorador de archivos local (para elegir rutas del YAML)
- `GET /api/explorar?ruta=<dir>` →
  `{"ruta": "/abs", "padre": "/abs/..", "dirs": [...], "archivos": [...]}`
  Úsalo como modal selector de archivos (navegar carpetas, elegir un `.BIN`,
  `.mp4`, etc.); al elegir, se inserta la ruta en el campo correspondiente.

## Estructura de la app (páginas)

Barra lateral con: **Flujo de trabajo**, **Configuraciones**, **Trabajos**,
**Resultados**, **Ajustes**. Cabecera con indicador de conexión (verde/rojo,
versión del servidor) y el nombre de la config activa (selector global).

### 1) Flujo de trabajo (página principal)
Un asistente vertical de 5 pasos (stepper) sobre la config activa:

- **Paso 0 · Selftest**: botón "Ejecutar selftest", estado del último
  (✓ TODO OK / ✗ con enlace al log). Texto de ayuda: "valida la matemática sin
  datos; si falla, el problema es del entorno Python, no de tus datos".
- **Paso 1 · Inspeccionar**: botón que lanza `inspeccionar` con la config
  activa y muestra el log resultante (es una tabla de texto con mensajes del
  log y sus frecuencias). Ayuda: "con esto decides dataset.imu.mensaje
  (IMU o GYR+ACC), instancia y gt.fuente en la configuración".
- **Paso 2 · Extraer frames**: botón que lanza `extraer-frames`. Ayuda: "solo
  hace falta una vez por vídeo; si los frames ya existen, se reutilizan".
- **Paso 3 · Offset temporal**: formulario con `t_ini` (s), `t_fin` (s),
  `rango` (s, por defecto 2.0) y botón "Estimar". Al terminar, mostrar el
  offset en ms grande y claro, el aviso de contraste si procede, y un botón
  **"Aplicar a la configuración"** que llama a
  `POST /api/configs/{config}/campo` con
  `ruta = "dataset.camara.offset_temporal_s"` y `valor = offset_s`, avisando
  de que el YAML se reescribe sin comentarios. Ayuda: "elige un tramo del
  vuelo con giros claros".
- **Paso 4 · Ejecutar**: checkbox "sin dead reckoning (más rápido)" y botón
  "Ejecutar pipeline". Al terminar en `ok`, mostrar tarjetas con
  `ate_se3`, `ate_sim3` y `aceptadas_pct`, y un enlace directo a la página de
  Resultados del `directorio` generado.

Cada paso, al lanzarse, muestra su log en vivo embebido (colapsable). Los
pasos no se bloquean entre sí (el usuario puede saltar), pero se marca
visualmente cuál fue el último completado.

Regla de semáforo para `aceptadas_pct` en el paso 4: verde ≥ 80, ámbar 50–80,
rojo < 50 con el texto "revisa offset temporal, extrínseca T_body_cam o
unidades de la IMU".

### 2) Configuraciones
- Lista de YAMLs; al seleccionar, **editor de texto** con resaltado YAML,
  números de línea, botón Guardar (PUT) y errores 422 mostrados en contexto.
- Junto al editor, un panel plegable "campos frecuentes" que lee de
  `contenido` y muestra (solo lectura, como referencia rápida):
  `dataset.tipo`, `dataset.log`, `dataset.camara.video`,
  `dataset.camara.timestamps`, `dataset.camara.fps`, `dataset.camara.t0`,
  `dataset.camara.offset_temporal_s`, `camara.modelo`, `camara.intrinsecos`,
  `eskf.modo`, `inicializacion.tipo`. Si alguno no existe, mostrar "—".
- Botón "insertar ruta…" que abre el explorador (`/api/explorar`) y pega la
  ruta elegida en el cursor del editor.
- No hay endpoint de crear/borrar configs: para crear, el usuario duplica
  guardando con otro nombre — añade un botón "Guardar como…" que hace PUT a
  `/api/configs/{nuevo_nombre.yaml}` con el texto actual.

### 3) Trabajos
Tabla de todos los trabajos (sondear `GET /api/trabajos` cada 2 s si hay
alguno vivo): comando, config, estado con badge de color, duración
(fin−inicio o "en curso"), y el comando exacto (`cmd`) en tooltip. Al hacer
clic, panel de detalle con el log completo (sondeo incremental con `desde`).
No existe endpoint para cancelar trabajos: no inventes ese botón.

### 4) Resultados
Tarjetas por informe (nombre, fecha de `modificado`). Detalle:
- Galería con `trayectoria.png`, `filtro.png`, `frontend.png` (grandes, con
  lightbox).
- `resumen` en un bloque monoespaciado.
- Lista del resto de archivos con botón de descarga
  (`est.tum`, `gt.tum`, `resultado.npz`).
Botón "Actualizar" (no hace falta sondeo aquí).

### 5) Ajustes
- URL base de la API (localStorage) + botón probar conexión.
- Info del servidor (`raiz`, `version`, `pymavlink`).

## Detalles de comportamiento

- Los trabajos largos siguen vivos aunque el usuario cambie de página: el
  sondeo pertenece a un store global, no al componente.
- Deshabilitar el botón de lanzar un comando mientras haya un trabajo del
  mismo comando `en_cola`/`corriendo` (evitar dobles clics), con el texto
  "ya hay un trabajo de este tipo en curso".
- Errores HTTP: mostrar `detail` del cuerpo si existe; si no, el status.
- Tiempos (`inicio`, `fin`, `modificado`) llegan como epoch en segundos:
  formatear a hora local.
- Nada de datos simulados: si la API no responde, se ve el estado vacío/error,
  no placeholders con datos falsos.

## Estilo

Tema oscuro técnico (panel de telemetría de dron): fondo gris muy oscuro,
acento único (ámbar o cian), tipografía monoespaciada para logs/YAML/rutas y
una sans limpia para el resto. Densidad media, sin ilustraciones decorativas.
Badges de estado: en_cola gris, corriendo azul animado, ok verde, error rojo.
Responsive básico (uso principal en escritorio).
