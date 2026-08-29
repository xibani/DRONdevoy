import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { useAtalaya } from "@/lib/atalaya-store";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const Route = createFileRoute("/ajustes")({
  head: () => ({
    meta: [
      { title: "Ajustes — ATALAYA VIO" },
      {
        name: "description",
        content:
          "Configura la URL base de la API de ATALAYA y consulta la información del servidor local.",
      },
      { property: "og:title", content: "Ajustes — ATALAYA VIO" },
      {
        property: "og:description",
        content: "URL base de la API e información del servidor local.",
      },
    ],
  }),
  component: Ajustes,
});

function Ajustes() {
  const { baseUrl, cambiarBaseUrl, estado } = useAtalaya();
  const [valor, setValor] = useState(baseUrl);
  const [probando, setProbando] = useState(false);

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-semibold">Ajustes</h1>

      <section className="space-y-3 rounded-lg border border-border bg-card p-4">
        <Label className="font-mono text-xs">URL base de la API</Label>
        <Input
          className="font-mono"
          value={valor}
          onChange={(e) => setValor(e.target.value)}
        />
        <div className="flex gap-2">
          <Button
            onClick={() => {
              cambiarBaseUrl(valor);
              toast.success("URL guardada");
            }}
          >
            Guardar
          </Button>
          <Button
            variant="outline"
            disabled={probando}
            onClick={async () => {
              setProbando(true);
              try {
                const e = await api.estado(valor.replace(/\/+$/, ""));
                toast.success(`Conexión correcta · v${e.version}`);
              } catch (err) {
                toast.error((err as Error).message);
              } finally {
                setProbando(false);
              }
            }}
          >
            Probar conexión
          </Button>
        </div>
      </section>

      <section className="space-y-2 rounded-lg border border-border bg-card p-4 font-mono text-sm">
        <h2 className="font-sans font-semibold">Información del servidor</h2>
        <p>
          <span className="text-muted-foreground">version:</span>{" "}
          {estado?.version ?? "—"}
        </p>
        <p className="break-all">
          <span className="text-muted-foreground">raiz:</span> {estado?.raiz ?? "—"}
        </p>
        <p>
          <span className="text-muted-foreground">pymavlink:</span>{" "}
          <span className={estado?.pymavlink ? "text-exito" : "text-destructive"}>
            {estado ? String(estado.pymavlink) : "—"}
          </span>
        </p>
        <p>
          <span className="text-muted-foreground">configs:</span>{" "}
          {estado?.configs ?? "—"}
        </p>
      </section>
    </div>
  );
}
