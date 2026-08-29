Por qué funciona
Porque las tres piezas hacen exactamente una cosa cada una y se comunican por un único canal:

atalaya/ tiene toda la lógica (ESKF, front-end, evaluación) y se ejecuta siempre como CLI: python -m atalaya <comando>. Nadie re-implementa nada de esto.

servidor_api.py no calcula nada: cuando la GUI pide un comando (POST /api/trabajos), lanza ese mismo CLI como subproceso y va capturando su stdout en memoria. La GUI sondea GET /api/trabajos/{id}?desde=N cada segundo y recibe solo los caracteres nuevos del log — por eso lo que ves en la web es exactamente lo que verías en la terminal. Los YAML y los informes de results/ se sirven como texto/archivos tal cual.

gui/ es un cliente tonto: todo pasa por api.ts (una función por endpoint, URL base de localStorage) y el sondeo vive en un store global, así que un trabajo sigue vivo aunque cambies de página.

La sesión anterior arregló las tres cosas que impedían que esto encajara en tu máquina: el log perdía las últimas líneas por una carrera en el sondeo, los acentos llegaban rotos (el subproceso emitía UTF-8 y el servidor decodificaba cp1252 — típico de Windows), y el enlace a resultados fallaba con rutas con backslashes.

Un detalle clave de tu entorno: el python del PATH (anaconda base) no tiene cv2, así que hay que usar el venv de vio-euroc, que es donde están todas las dependencias (y donde instalé fastapi/uvicorn).

Primera ejecución, paso a paso
1. Arrancar las dos piezas (dos terminales):

# Teriminal
```python
# Terminal 1 — la API
cd c:\Users\ander\Documents\00-Proyectos\DRONdevoy\atalaya_fw
& "..\vio-euroc\.venv\Scripts\python.exe" servidor_api.py

# Terminal 2 — la GUI
cd c:\Users\ander\Documents\00-Proyectos\DRONdevoy\atalaya_fw\gui
npm run dev
```

Abre http://localhost:8080. Si la cabecera dice «conectado · v0.1.0», la GUI ya habla con el servidor. (Si algún día cambias el puerto o la máquina de la API, se edita en Ajustes.)

2. Selftest (paso 0). Botón «Ejecutar selftest» en la página principal. No necesita datos: valida la matemática del paquete. Debe terminar en «TODO OK». Si falla, el problema es del entorno Python, no de tus datos.

3. Preparar una configuración. En Configuraciones, parte de ardupilot_ejemplo.yaml y guárdala con otro nombre («Guardar como…», p. ej. mi_vuelo.yaml). Lo mínimo que hay que rellenar: la ruta del log .BIN de ArduPilot y la del vídeo de la cámara — el botón «Insertar ruta…» abre un explorador del disco para pegarlas sin escribirlas a mano. Ojo: ahora mismo la GUI te muestra el aviso de que pymavlink no está instalado; para leer logs de ArduPilot hace falta antes:


& "..\vio-euroc\.venv\Scripts\python.exe" -m pip install pymavlink
(y reiniciar el servidor para que el aviso desaparezca).

4. Inspeccionar (paso 1). Con mi_vuelo.yaml como config activa (selector de la cabecera), lanza «Inspeccionar log». El log del trabajo te dice qué mensajes trae el .BIN (qué IMUs y a qué Hz, si hay GPS/actitud). Con eso decides en el YAML dataset.imu.mensaje (IMU o GYR+ACC), la instancia y gt.fuente.

5. Extraer frames (paso 2). Convierte el vídeo en imágenes con timestamps. Solo hace falta una vez por vídeo; si ya existen, se reutilizan.

6. Offset temporal (paso 3). Elige un tramo del vuelo con giros claros (t_ini/t_fin en segundos) y pulsa «Estimar». Te da el desfase cámara–IMU en ms; si avisa de «poca rotación», elige otro tramo. Con un valor fiable, «Aplicar a la configuración» lo escribe en el YAML (aviso: esa escritura reescribe el archivo y pierde los comentarios).

7. Ejecutar (paso 4). «Ejecutar pipeline» corre el proceso completo — puede tardar minutos, el log va en vivo y puedes navegar por otras páginas mientras. Al terminar verás las tarjetas de métricas (ate_se3, ate_sim3, % aceptadas con semáforo: verde ≥ 80 %) y un enlace al informe en Resultados: gráficas de trayectoria/filtro/frontend, el resumen.txt y los archivos descargables (est.tum, resultado.npz…).

Si no tienes aún un vuelo propio, la alternativa es euroc_mh01.yaml, pero requiere tener descargado el dataset EuRoC MH01 en la ruta que indica ese YAML — no está en el repo.