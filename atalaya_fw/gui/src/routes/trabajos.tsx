import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { useAtalaya } from "@/lib/atalaya-store";
import { duracion, fechaLocal } from "@/lib/api";
import { BadgeEstado } from "@/components/BadgeEstado";
import { TerminalLog } from "@/components/TerminalLog";

export const Route = createFileRoute("/trabajos")({
  validateSearch: (search: Record<string, unknown>) => ({
    id: typeof search["id"] === "string" ? (search["id"] as string) : undefined,
  }),
  head: () => ({
    meta: [
      { title: "Trabajos — ATALAYA VIO" },
      {
        name: "description",
        content:
          "Historial y seguimiento en vivo de los trabajos del pipeline VIO: estado, duración y log completo.",
      },
      { property: "og:title", content: "Trabajos — ATALAYA VIO" },
      {
        property: "og:description",
        content: "Estado, duración y log completo de cada trabajo del pipeline.",
      },
    ],
  }),
  component: Trabajos,
});

function Trabajos() {
  const { trabajos, logs, seguirLog } = useAtalaya();
  const { id } = Route.useSearch();
  const navigate = useNavigate();
  const seleccionado = trabajos.find((t) => t.id === id);

  useEffect(() => {
    if (id) seguirLog(id);
  }, [id, seguirLog]);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Trabajos</h1>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-card font-mono text-xs text-muted-foreground">
            <tr>
              <th className="p-2 text-left">comando</th>
              <th className="p-2 text-left">config</th>
              <th className="p-2 text-left">estado</th>
              <th className="p-2 text-left">inicio</th>
              <th className="p-2 text-left">duración</th>
            </tr>
          </thead>
          <tbody>
            {trabajos.length === 0 && (
              <tr>
                <td colSpan={5} className="p-6 text-center text-muted-foreground">
                  No hay trabajos todavía.
                </td>
              </tr>
            )}
            {trabajos.map((t) => (
              <tr
                key={t.id}
                title={t.cmd ?? ""}
                onClick={() => navigate({ to: "/trabajos", search: { id: t.id } })}
                className={`cursor-pointer border-t border-border hover:bg-accent/50 ${
                  t.id === id ? "bg-accent/60" : ""
                }`}
              >
                <td className="p-2 font-mono">{t.comando}</td>
                <td className="p-2 font-mono text-xs text-muted-foreground">
                  {t.config ?? "—"}
                </td>
                <td className="p-2">
                  <BadgeEstado estado={t.estado} />
                </td>
                <td className="p-2 font-mono text-xs">{fechaLocal(t.inicio)}</td>
                <td className="p-2 font-mono text-xs">{duracion(t)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {seleccionado && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-semibold">Detalle</h2>
            <span className="font-mono text-xs text-muted-foreground">
              {seleccionado.id}
            </span>
            <BadgeEstado estado={seleccionado.estado} />
            {seleccionado.codigo !== undefined && (
              <span className="font-mono text-xs text-muted-foreground">
                código {seleccionado.codigo}
              </span>
            )}
          </div>
          {seleccionado.cmd && (
            <pre className="overflow-x-auto rounded border border-border bg-card p-2 font-mono text-xs text-muted-foreground">
              {seleccionado.cmd}
            </pre>
          )}
          <TerminalLog
            texto={logs[seleccionado.id]?.texto ?? ""}
            altura="h-96"
            titulo="log completo"
          />
        </div>
      )}
    </div>
  );
}
