import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  guardarBaseUrl,
  leerBaseUrl,
  URL_POR_DEFECTO,
  type Comando,
  type Estado,
  type Trabajo,
} from "./api";

type Logs = Record<string, { texto: string; total: number }>;

type Ctx = {
  baseUrl: string;
  cambiarBaseUrl: (u: string) => void;
  estado: Estado | null;
  errorEstado: string | null;
  cargandoEstado: boolean;
  recomprobar: () => void;
  configs: string[];
  configActiva: string | null;
  setConfigActiva: (c: string) => void;
  recargarConfigs: () => void;
  trabajos: Trabajo[];
  logs: Logs;
  seguirLog: (id: string) => void;
  lanzar: (comando: Comando, extra?: Record<string, unknown>) => Promise<string>;
  ocupado: (comando: Comando) => boolean;
  ultimo: (comando: Comando) => Trabajo | undefined;
};

const AtalayaCtx = createContext<Ctx | null>(null);
const CLAVE_CONFIG = "atalaya.config_activa";

export function AtalayaProvider({ children }: { children: ReactNode }) {
  const [baseUrl, setBaseUrl] = useState(URL_POR_DEFECTO);
  const [hidratado, setHidratado] = useState(false);
  const [estado, setEstado] = useState<Estado | null>(null);
  const [errorEstado, setErrorEstado] = useState<string | null>(null);
  const [cargandoEstado, setCargandoEstado] = useState(true);
  const [configs, setConfigs] = useState<string[]>([]);
  const [configActiva, setConfigActivaEstado] = useState<string | null>(null);
  const [trabajos, setTrabajos] = useState<Trabajo[]>([]);
  const [logs, setLogs] = useState<Logs>({});
  const seguidos = useRef<Set<string>>(new Set());
  const completos = useRef<Set<string>>(new Set()); // log ya completo (estado terminal)
  const logsRef = useRef<Logs>({});
  const [tic, setTic] = useState(0);
  const [ticTrabajos, setTicTrabajos] = useState(0);

  useEffect(() => {
    logsRef.current = logs;
  }, [logs]);

  useEffect(() => {
    setBaseUrl(leerBaseUrl());
    setConfigActivaEstado(window.localStorage.getItem(CLAVE_CONFIG));
    setHidratado(true);
  }, []);

  const cambiarBaseUrl = useCallback((u: string) => {
    const limpia = u.replace(/\/+$/, "");
    guardarBaseUrl(limpia);
    setBaseUrl(limpia);
    // los trabajos y logs pertenecen al servidor anterior
    seguidos.current.clear();
    completos.current.clear();
    setLogs({});
    setTrabajos([]);
    setTic((t) => t + 1);
    setTicTrabajos((t) => t + 1);
  }, []);

  const setConfigActiva = useCallback((c: string) => {
    window.localStorage.setItem(CLAVE_CONFIG, c);
    setConfigActivaEstado(c);
  }, []);

  const cargarConfigs = useCallback(
    async (base: string) => {
      try {
        const lista = await api.configs(base);
        setConfigs(lista);
        setConfigActivaEstado((actual) =>
          actual && lista.includes(actual) ? actual : (lista[0] ?? null),
        );
      } catch {
        setConfigs([]);
      }
    },
    [],
  );

  // Estado del servidor
  useEffect(() => {
    if (!hidratado) return;
    let vivo = true;
    setCargandoEstado(true);
    api
      .estado(baseUrl)
      .then((e) => {
        if (!vivo) return;
        setEstado(e);
        setErrorEstado(null);
        void cargarConfigs(baseUrl);
      })
      .catch((e: Error) => {
        if (!vivo) return;
        setEstado(null);
        setErrorEstado(e.message);
      })
      .finally(() => vivo && setCargandoEstado(false));
    return () => {
      vivo = false;
    };
  }, [baseUrl, hidratado, tic, cargarConfigs]);

  // Sondeo global de trabajos: cada ~1,5 s mientras haya trabajos vivos;
  // se detiene del todo cuando no los hay (lanzar() lo vuelve a despertar).
  useEffect(() => {
    if (!estado) return;
    let vivo = true;
    let timer: ReturnType<typeof setTimeout>;
    let habiaActivos = false;
    const tick = async () => {
      try {
        const lista = await api.trabajos(baseUrl);
        if (!vivo) return;
        setTrabajos(lista);
        habiaActivos = lista.some(
          (t) => t.estado === "en_cola" || t.estado === "corriendo",
        );
        if (habiaActivos) timer = setTimeout(tick, 1500);
      } catch {
        // error transitorio: reintenta solo si seguíamos un trabajo vivo
        if (vivo && habiaActivos) timer = setTimeout(tick, 5000);
      }
    };
    void tick();
    return () => {
      vivo = false;
      clearTimeout(timer);
    };
  }, [baseUrl, estado, ticTrabajos]);

  // Sondeo incremental de logs de los trabajos seguidos (cada 1 s, con
  // `desde` = caracteres ya recibidos). El fin del log lo marca el `estado`
  // de la PROPIA respuesta: así nunca se pierde la cola aunque la tabla de
  // trabajos ya haya visto el estado terminal.
  useEffect(() => {
    if (!estado) return;
    let vivo = true;
    const timer = setInterval(async () => {
      for (const id of Array.from(seguidos.current)) {
        if (completos.current.has(id)) continue;
        try {
          const det = await api.trabajo(baseUrl, id, logsRef.current[id]?.total ?? 0);
          if (!vivo) return;
          setLogs((prev) => {
            const anterior = prev[id]?.texto ?? "";
            return {
              ...prev,
              [id]: { texto: anterior + (det.log ?? ""), total: det.log_total ?? 0 },
            };
          });
          if (det.estado === "ok" || det.estado === "error") {
            completos.current.add(id);
          }
        } catch {
          /* reintenta en el siguiente ciclo */
        }
      }
    }, 1000);
    return () => {
      vivo = false;
      clearInterval(timer);
    };
  }, [baseUrl, estado]);

  const seguirLog = useCallback((id: string) => {
    seguidos.current.add(id);
  }, []);

  const lanzar = useCallback(
    async (comando: Comando, extra: Record<string, unknown> = {}) => {
      const cuerpo: Record<string, unknown> = { comando, ...extra };
      if (comando !== "selftest") cuerpo["config"] = configActiva;
      const { id } = await api.crearTrabajo(baseUrl, cuerpo);
      seguidos.current.add(id);
      setLogs((p) => ({ ...p, [id]: { texto: "", total: 0 } }));
      setTicTrabajos((t) => t + 1); // despierta el sondeo de la tabla
      return id;
    },
    [baseUrl, configActiva],
  );

  const ocupado = useCallback(
    (comando: Comando) =>
      trabajos.some(
        (t) =>
          t.comando === comando &&
          (t.estado === "en_cola" || t.estado === "corriendo"),
      ),
    [trabajos],
  );

  const ultimo = useCallback(
    (comando: Comando) =>
      [...trabajos]
        .filter((t) => t.comando === comando)
        .sort((a, b) => (b.inicio ?? 0) - (a.inicio ?? 0))[0],
    [trabajos],
  );

  const valor = useMemo<Ctx>(
    () => ({
      baseUrl,
      cambiarBaseUrl,
      estado,
      errorEstado,
      cargandoEstado,
      recomprobar: () => setTic((t) => t + 1),
      configs,
      configActiva,
      setConfigActiva,
      recargarConfigs: () => void cargarConfigs(baseUrl),
      trabajos,
      logs,
      seguirLog,
      lanzar,
      ocupado,
      ultimo,
    }),
    [
      baseUrl,
      cambiarBaseUrl,
      estado,
      errorEstado,
      cargandoEstado,
      configs,
      configActiva,
      setConfigActiva,
      cargarConfigs,
      trabajos,
      logs,
      seguirLog,
      lanzar,
      ocupado,
      ultimo,
    ],
  );

  return <AtalayaCtx.Provider value={valor}>{children}</AtalayaCtx.Provider>;
}

export function useAtalaya() {
  const ctx = useContext(AtalayaCtx);
  if (!ctx) throw new Error("useAtalaya fuera del proveedor");
  return ctx;
}
