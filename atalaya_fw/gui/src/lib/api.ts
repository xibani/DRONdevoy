export const URL_POR_DEFECTO = "http://localhost:8420";
const CLAVE = "atalaya.api_url";

export function leerBaseUrl(): string {
  if (typeof window === "undefined") return URL_POR_DEFECTO;
  return window.localStorage.getItem(CLAVE) || URL_POR_DEFECTO;
}

export function guardarBaseUrl(url: string) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(CLAVE, url.replace(/\/+$/, ""));
}

export type Estado = {
  ok: boolean;
  version: string;
  raiz: string;
  pymavlink: boolean;
  configs: number;
};

export type EstadoTrabajo = "en_cola" | "corriendo" | "ok" | "error";

export type Trabajo = {
  id: string;
  comando: Comando;
  config: string | null;
  estado: EstadoTrabajo;
  resultado: Record<string, unknown> | null;
  inicio: number | null;
  fin: number | null;
  cmd?: string;
  codigo?: number;
};

export type TrabajoDetalle = Trabajo & { log: string; log_total: number };

export type Comando =
  | "selftest"
  | "inspeccionar"
  | "extraer-frames"
  | "offset"
  | "ejecutar";

export type ConfigArchivo = {
  nombre: string;
  texto: string;
  contenido: Record<string, unknown>;
};

export type Resultado = {
  nombre: string;
  archivos: string[];
  resumen: string | null;
  modificado: number;
};

export type Exploracion = {
  ruta: string;
  padre: string | null;
  dirs: string[];
  archivos: string[];
};

export class ErrorApi extends Error {}

async function pedir<T>(base: string, ruta: string, init?: RequestInit): Promise<T> {
  let res: Response;
  const opciones: RequestInit = { ...init };
  if (init?.body) opciones.headers = { "Content-Type": "application/json" };
  try {
    res = await fetch(`${base}${ruta}`, opciones);
  } catch {
    throw new ErrorApi("No se pudo contactar con el servidor");
  }
  if (!res.ok) {
    let detalle = `Error ${res.status}`;
    try {
      const cuerpo = await res.json();
      if (cuerpo && typeof cuerpo.detail === "string") detalle = cuerpo.detail;
    } catch {
      /* sin cuerpo JSON */
    }
    throw new ErrorApi(detalle);
  }
  return (await res.json()) as T;
}

export const api = {
  estado: (b: string) => pedir<Estado>(b, "/api/estado"),
  configs: (b: string) => pedir<string[]>(b, "/api/configs"),
  config: (b: string, n: string) =>
    pedir<ConfigArchivo>(b, `/api/configs/${encodeURIComponent(n)}`),
  guardarConfig: (b: string, n: string, texto: string) =>
    pedir<unknown>(b, `/api/configs/${encodeURIComponent(n)}`, {
      method: "PUT",
      body: JSON.stringify({ texto }),
    }),
  campoConfig: (b: string, n: string, ruta: string, valor: unknown) =>
    pedir<unknown>(b, `/api/configs/${encodeURIComponent(n)}/campo`, {
      method: "POST",
      body: JSON.stringify({ ruta, valor }),
    }),
  trabajos: (b: string) => pedir<Trabajo[]>(b, "/api/trabajos"),
  trabajo: (b: string, id: string, desde: number) =>
    pedir<TrabajoDetalle>(b, `/api/trabajos/${id}?desde=${desde}`),
  crearTrabajo: (b: string, cuerpo: Record<string, unknown>) =>
    pedir<{ id: string }>(b, "/api/trabajos", {
      method: "POST",
      body: JSON.stringify(cuerpo),
    }),
  resultados: (b: string) => pedir<Resultado[]>(b, "/api/resultados"),
  explorar: (b: string, ruta?: string) =>
    pedir<Exploracion>(
      b,
      `/api/explorar${ruta ? `?ruta=${encodeURIComponent(ruta)}` : ""}`,
    ),
  urlArchivo: (b: string, nombre: string, fname: string) =>
    `${b}/api/resultados/${encodeURIComponent(nombre)}/archivo/${encodeURIComponent(fname)}`,
};

export function fechaLocal(epoch: number | null | undefined) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString("es-ES");
}

export function duracion(t: Trabajo) {
  if (!t.inicio) return "—";
  if (!t.fin) return "en curso";
  const s = t.fin - t.inicio;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}
