import type { EstadoTrabajo } from "@/lib/api";
import { cn } from "@/lib/utils";

const mapa: Record<EstadoTrabajo, { texto: string; clase: string }> = {
  en_cola: { texto: "en cola", clase: "bg-muted text-muted-foreground border-border" },
  corriendo: {
    texto: "corriendo",
    clase: "bg-info/15 text-info border-info/40 animate-pulse",
  },
  ok: { texto: "ok", clase: "bg-exito/15 text-exito border-exito/40" },
  error: {
    texto: "error",
    clase: "bg-destructive/15 text-destructive border-destructive/40",
  },
};

export function BadgeEstado({ estado }: { estado: EstadoTrabajo }) {
  const e = mapa[estado] ?? mapa.en_cola;
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-2 py-0.5 font-mono text-xs",
        e.clase,
      )}
    >
      {e.texto}
    </span>
  );
}
