import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useAtalaya } from "@/lib/atalaya-store";
import { api, type ConfigArchivo } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ExploradorArchivos } from "@/components/ExploradorArchivos";
import { ChevronDown } from "lucide-react";

export const Route = createFileRoute("/configuraciones")({
  head: () => ({
    meta: [
      { title: "Configuraciones — ATALAYA VIO" },
      {
        name: "description",
        content:
          "Edita los archivos YAML de configuración del pipeline VIO: dataset, cámara, ESKF e inicialización.",
      },
      { property: "og:title", content: "Configuraciones — ATALAYA VIO" },
      {
        property: "og:description",
        content: "Editor YAML con campos frecuentes y explorador de rutas locales.",
      },
    ],
  }),
  component: Configuraciones,
});

const CAMPOS = [
  "dataset.tipo",
  "dataset.log",
  "dataset.camara.video",
  "dataset.camara.timestamps",
  "dataset.camara.fps",
  "dataset.camara.t0",
  "dataset.camara.offset_temporal_s",
  "camara.modelo",
  "camara.intrinsecos",
  "eskf.modo",
  "inicializacion.tipo",
];

function leerRuta(obj: unknown, ruta: string): string {
  let actual: unknown = obj;
  for (const parte of ruta.split(".")) {
    if (actual && typeof actual === "object" && parte in (actual as object)) {
      actual = (actual as Record<string, unknown>)[parte];
    } else {
      return "—";
    }
  }
  if (actual === null || actual === undefined) return "—";
  return typeof actual === "object" ? JSON.stringify(actual) : String(actual);
}

function Configuraciones() {
  const { baseUrl, configs, configActiva, setConfigActiva, recargarConfigs } =
    useAtalaya();
  const [datos, setDatos] = useState<ConfigArchivo | null>(null);
  const [texto, setTexto] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [guardando, setGuardando] = useState(false);
  const [explorador, setExplorador] = useState(false);
  const [camposAbierto, setCamposAbierto] = useState(true);
  const [nuevoNombre, setNuevoNombre] = useState("");
  const areaRef = useRef<HTMLTextAreaElement>(null);

  const cargar = useCallback(
    async (nombre: string) => {
      try {
        setError(null);
        const d = await api.config(baseUrl, nombre);
        setDatos(d);
        setTexto(d.texto);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [baseUrl],
  );

  useEffect(() => {
    if (configActiva) void cargar(configActiva);
  }, [configActiva, cargar]);

  const guardar = async (nombre: string) => {
    setGuardando(true);
    try {
      await api.guardarConfig(baseUrl, nombre, texto);
      setError(null);
      toast.success(`Guardado ${nombre}`);
      recargarConfigs();
      await cargar(nombre);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setGuardando(false);
    }
  };

  const insertar = (ruta: string) => {
    const area = areaRef.current;
    if (!area) return;
    const pos = area.selectionStart;
    setTexto(texto.slice(0, pos) + ruta + texto.slice(area.selectionEnd));
  };

  const lineas = texto.split("\n").length;

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Configuraciones</h1>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[220px_1fr]">
        <div className="space-y-1">
          {configs.length === 0 && (
            <p className="text-sm text-muted-foreground">No hay configuraciones.</p>
          )}
          {configs.map((c) => (
            <button
              key={c}
              onClick={() => setConfigActiva(c)}
              className={`w-full truncate rounded-md border border-border px-3 py-2 text-left font-mono text-xs hover:border-primary/60 ${
                c === configActiva ? "border-primary/60 bg-primary/10 text-primary" : ""
              }`}
            >
              {c}
            </button>
          ))}
        </div>

        <div className="min-w-0 space-y-3">
          {datos && (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  disabled={guardando}
                  onClick={() => void guardar(datos.nombre)}
                >
                  Guardar
                </Button>
                <Button variant="outline" onClick={() => setExplorador(true)}>
                  Insertar ruta…
                </Button>
                <Input
                  className="w-56 font-mono text-xs"
                  placeholder="nuevo_nombre.yaml"
                  value={nuevoNombre}
                  onChange={(e) => setNuevoNombre(e.target.value)}
                />
                <Button
                  variant="outline"
                  disabled={!nuevoNombre.trim() || guardando}
                  onClick={async () => {
                    const n = nuevoNombre.trim();
                    await guardar(n.endsWith(".yaml") ? n : `${n}.yaml`);
                    setNuevoNombre("");
                  }}
                >
                  Guardar como…
                </Button>
              </div>

              <div className="flex overflow-hidden rounded-md border border-border bg-terminal">
                <pre className="min-w-10 shrink-0 border-r border-border px-2 py-3 text-right font-mono text-xs leading-6 text-muted-foreground select-none">
                  {Array.from({ length: lineas }, (_, i) => i + 1).join("\n")}
                </pre>
                <textarea
                  ref={areaRef}
                  value={texto}
                  onChange={(e) => setTexto(e.target.value)}
                  spellCheck={false}
                  className="h-[28rem] flex-1 resize-none bg-transparent p-3 py-3 font-mono text-xs leading-6 text-terminal-foreground outline-none"
                />
              </div>

              {error && (
                <p className="rounded border border-destructive/40 bg-destructive/10 p-2 font-mono text-xs text-destructive">
                  {error}
                </p>
              )}

              <div className="rounded-md border border-border bg-card">
                <button
                  className="flex w-full items-center gap-2 px-3 py-2 text-sm"
                  onClick={() => setCamposAbierto((a) => !a)}
                >
                  <ChevronDown
                    className={`size-4 transition-transform ${camposAbierto ? "rotate-180" : ""}`}
                  />
                  Campos frecuentes (solo lectura)
                </button>
                {camposAbierto && (
                  <dl className="grid grid-cols-1 gap-x-6 gap-y-1 px-4 pb-3 font-mono text-xs sm:grid-cols-2">
                    {CAMPOS.map((c) => (
                      <div key={c} className="flex justify-between gap-3 truncate">
                        <dt className="text-muted-foreground">{c}</dt>
                        <dd className="truncate">{leerRuta(datos.contenido, c)}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <ExploradorArchivos
        abierto={explorador}
        onAbrir={setExplorador}
        onElegir={insertar}
      />
    </div>
  );
}
