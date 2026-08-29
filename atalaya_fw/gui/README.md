# ATALAYA — interfaz web

Panel de control del pipeline VIO. No tiene lógica propia: todo pasa por la
API REST de `servidor_api.py` (puerto 8420), que es la única fuente de verdad.
Toda la comunicación HTTP vive en [src/lib/api.ts](src/lib/api.ts) (una función
por endpoint real, con sus tipos) y el estado global (conexión, trabajos,
sondeo de logs) en [src/lib/atalaya-store.tsx](src/lib/atalaya-store.tsx).

Stack: React 19 + TanStack Start (router con SSR) + Vite 8 + Tailwind 4 +
shadcn/ui. Generado con Lovable a partir de [PROMPT_LOVABLE.md](PROMPT_LOVABLE.md).

## Desarrollo

```sh
# terminal 1 — la API (desde la raíz del repo, con el Python que tenga
# instalado atalaya: cv2, fastapi, uvicorn)
python servidor_api.py

# terminal 2 — la GUI
cd gui
npm install
npm run dev          # http://localhost:8080
```

No hay proxy de desarrollo: la GUI habla directamente con la API usando la
URL base guardada en `localStorage` (por defecto `http://localhost:8420`,
editable en **Ajustes**). El servidor ya tiene CORS abierto.

## Producción

```sh
cd gui
npm run build        # genera .output/ (SSR con nitro)
npx vite preview     # sirve la build localmente
```

## Comportamiento clave (contrato con el servidor)

- Los comandos se lanzan con `POST /api/trabajos` y se sondean con
  `GET /api/trabajos/{id}?desde=N` cada 1 s de forma incremental (`desde` =
  caracteres ya recibidos; se acumula `log` y se avanza con `log_total`).
  El sondeo vive en el store global: sobrevive a cambios de página.
- El fin del log lo marca el `estado` de la propia respuesta del detalle
  (`ok`/`error`), no la tabla de trabajos: así nunca se pierde la cola del log.
- La tabla de trabajos se sondea cada ~1,5 s solo mientras haya trabajos
  `en_cola`/`corriendo`; sin trabajos vivos no hay tráfico (lanzar un trabajo
  vuelve a despertar el sondeo).
- Los YAML se editan como texto (`GET`/`PUT /api/configs/{nombre}`); un 422
  con `detail` («YAML inválido: …») se muestra bajo el editor. «Aplicar a la
  configuración» usa `POST /api/configs/{nombre}/campo` con
  `ruta = dataset.camara.offset_temporal_s` y avisa de que reescribe el YAML
  sin comentarios.
- Las imágenes de resultados se enlazan directas a
  `GET {base}/api/resultados/{nombre}/archivo/{fname}` (URL base incluida).
- Sin servidor: pantalla «Servidor no disponible» con las instrucciones de
  arranque y botón Reintentar. Errores HTTP: se muestra el `detail` si existe.

## Decisiones tomadas

- **Ningún endpoint inventado.** La GUI solo usa los endpoints reales de
  `servidor_api.py`. No hay botón de cancelar trabajos ni de borrar configs
  (no existen en el servidor); «crear config» se resuelve con «Guardar
  como…» (PUT a un nombre nuevo), que el servidor ya soporta.
- **Sin mocks residuales.** La GUI vino generada ya contra el contrato real;
  se auditó (grep de fixtures/MSW/latencias simuladas/arrays hardcodeados) y
  no quedaba ningún dato falso.
- **Arreglos aplicados sobre lo generado:**
  - El sondeo de logs podía perder las últimas líneas (dejaba de pedir en
    cuanto la *tabla* veía el estado terminal); ahora termina cuando la
    propia respuesta del detalle llega en estado terminal.
  - El sondeo de `GET /api/trabajos` no se detenía nunca (seguía cada 6 s);
    ahora para del todo cuando no hay trabajos vivos.
  - El nombre del informe se extraía de `resultado.directorio` con
    `split("/")`, que fallaba en Windows (barras invertidas y barra final).
  - Al cambiar la URL base se limpian trabajos/logs del servidor anterior.
- **Un cambio en `servidor_api.py`** (no toca `atalaya/`): el subproceso de
  los trabajos se captura ahora con `encoding="utf-8"` +
  `PYTHONIOENCODING=utf-8`; en Windows padre e hijo no compartían
  codificación y los acentos del log llegaban rotos a la GUI.
- El README original (prompt de generación con el contrato completo de la
  API) se conserva en [PROMPT_LOVABLE.md](PROMPT_LOVABLE.md).
