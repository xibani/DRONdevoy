import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import { useAtalaya } from "@/lib/atalaya-store";
import { api, fechaLocal, type Resultado } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Download, RefreshCw } from "lucide-react";

export const Route = createFileRoute("/resultados")({
  validateSearch: (search: Record<string, unknown>) => ({
    nombre:
      typeof search["nombre"] === "string" ? (search["nombre"] as string) : undefined,
  }),
  head: () => ({
    meta: [
      { title: "Resultados — ATALAYA VIO" },
      {
        name: "description",
        content:
          "Informes generados por el pipeline VIO: gráficas de trayectoria, filtro y frontend, resumen y archivos descargables.",
      },
      { property: "og:title", content: "Resultados — ATALAYA VIO" },
      {
        property: "og:description",
        content: "Gráficas de trayectoria, resumen y archivos de cada informe.",
      },
    ],
  }),
  component: Resultados,
});

const IMAGENES = ["trayectoria.png", "filtro.png", "frontend.png"];

function Resultados() {
  const { baseUrl } = useAtalaya();
  const { nombre } = Route.useSearch();
  const navigate = useNavigate();
  const [lista, setLista] = useState<Resultado[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);

  const cargar = useCallback(async () => {
    try {
      setError(null);
      setLista(await api.resultados(baseUrl));
    } catch (e) {
      setError((e as Error).message);
      setLista([]);
    }
  }, [baseUrl]);

  useEffect(() => {
    void cargar();
  }, [cargar]);

  const detalle = lista.find((r) => r.nombre === nombre);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold">Resultados</h1>
        <Button variant="outline" size="sm" onClick={() => void cargar()}>
          <RefreshCw className="size-4" /> Actualizar
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {!detalle && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {lista.length === 0 && !error && (
            <p className="text-sm text-muted-foreground">No hay informes generados.</p>
          )}
          {lista.map((r) => (
            <button
              key={r.nombre}
              onClick={() =>
                navigate({ to: "/resultados", search: { nombre: r.nombre } })
              }
              className="rounded-lg border border-border bg-card p-4 text-left transition-colors hover:border-primary/60"
            >
              <p className="font-mono font-semibold text-primary">{r.nombre}</p>
              <p className="mt-1 font-mono text-xs text-muted-foreground">
                {fechaLocal(r.modificado)} · {r.archivos.length} archivos
              </p>
            </button>
          ))}
        </div>
      )}

      {detalle && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                navigate({ to: "/resultados", search: { nombre: undefined } })
              }
            >
              ← Volver
            </Button>
            <h2 className="font-mono text-lg text-primary">{detalle.nombre}</h2>
            <span className="font-mono text-xs text-muted-foreground">
              {fechaLocal(detalle.modificado)}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {detalle.archivos
              .filter((f) => IMAGENES.includes(f))
              .map((f) => {
                const url = api.urlArchivo(baseUrl, detalle.nombre, f);
                return (
                  <figure key={f} className="rounded-lg border border-border bg-card p-2">
                    <img
                      src={url}
                      alt={`Gráfica ${f} del informe ${detalle.nombre}`}
                      className="w-full cursor-zoom-in rounded bg-terminal"
                      onClick={() => setLightbox(url)}
                    />
                    <figcaption className="mt-1 font-mono text-xs text-muted-foreground">
                      {f}
                    </figcaption>
                  </figure>
                );
              })}
          </div>

          {detalle.resumen && (
            <pre className="overflow-auto rounded-lg border border-border bg-terminal p-3 font-mono text-xs whitespace-pre-wrap text-terminal-foreground">
              {detalle.resumen}
            </pre>
          )}

          <div className="space-y-2">
            <h3 className="font-semibold">Archivos</h3>
            {detalle.archivos
              .filter((f) => !IMAGENES.includes(f))
              .map((f) => (
                <a
                  key={f}
                  href={api.urlArchivo(baseUrl, detalle.nombre, f)}
                  download
                  className="flex items-center gap-2 rounded border border-border bg-card px-3 py-2 font-mono text-sm hover:border-primary/60"
                >
                  <Download className="size-4 text-primary" /> {f}
                </a>
              ))}
          </div>
        </div>
      )}

      <Dialog open={!!lightbox} onOpenChange={(v) => !v && setLightbox(null)}>
        <DialogContent className="max-w-5xl">
          {lightbox && (
            <img src={lightbox} alt="Gráfica ampliada del informe" className="w-full" />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
