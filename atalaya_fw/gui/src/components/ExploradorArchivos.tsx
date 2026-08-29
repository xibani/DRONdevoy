import { useCallback, useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { api, type Exploracion } from "@/lib/api";
import { useAtalaya } from "@/lib/atalaya-store";
import { Folder, FileText, ArrowUp } from "lucide-react";

export function ExploradorArchivos({
  abierto,
  onAbrir,
  onElegir,
}: {
  abierto: boolean;
  onAbrir: (v: boolean) => void;
  onElegir: (ruta: string) => void;
}) {
  const { baseUrl } = useAtalaya();
  const [datos, setDatos] = useState<Exploracion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seleccion, setSeleccion] = useState<string | null>(null);

  const cargar = useCallback(
    async (ruta?: string) => {
      try {
        setError(null);
        setDatos(await api.explorar(baseUrl, ruta));
        setSeleccion(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [baseUrl],
  );

  useEffect(() => {
    if (abierto) void cargar();
  }, [abierto, cargar]);

  const unir = (base: string, hijo: string) =>
    `${base.replace(/\/+$/, "")}/${hijo}`;

  return (
    <Dialog open={abierto} onOpenChange={onAbrir}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Explorador de archivos</DialogTitle>
        </DialogHeader>
        <p className="truncate font-mono text-xs text-muted-foreground">
          {datos?.ruta ?? "…"}
        </p>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <div className="h-80 overflow-auto rounded-md border border-border bg-terminal p-1 font-mono text-sm">
          {datos?.padre && (
            <button
              className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-accent"
              onClick={() => void cargar(datos.padre!)}
            >
              <ArrowUp className="size-4 text-muted-foreground" /> ..
            </button>
          )}
          {datos?.dirs.map((d) => (
            <button
              key={d}
              className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-accent"
              onClick={() => void cargar(unir(datos.ruta, d))}
            >
              <Folder className="size-4 text-primary" /> {d}
            </button>
          ))}
          {datos?.archivos.map((f) => {
            const completa = unir(datos.ruta, f);
            return (
              <button
                key={f}
                className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-accent ${
                  seleccion === completa ? "bg-primary/15 text-primary" : ""
                }`}
                onClick={() => setSeleccion(completa)}
                onDoubleClick={() => {
                  onElegir(completa);
                  onAbrir(false);
                }}
              >
                <FileText className="size-4 text-muted-foreground" /> {f}
              </button>
            );
          })}
        </div>
        <DialogFooter className="gap-2">
          <Button
            variant="outline"
            onClick={() => {
              if (datos) {
                onElegir(datos.ruta);
                onAbrir(false);
              }
            }}
          >
            Usar carpeta actual
          </Button>
          <Button
            disabled={!seleccion}
            onClick={() => {
              if (seleccion) {
                onElegir(seleccion);
                onAbrir(false);
              }
            }}
          >
            Usar archivo
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
