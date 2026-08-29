import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { useAtalaya } from "@/lib/atalaya-store";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertTriangle, Radar, ListChecks, FileCode2, Images, Settings, Workflow } from "lucide-react";
import { cn } from "@/lib/utils";

const enlaces = [
  { to: "/", texto: "Flujo de trabajo", Icono: Workflow },
  { to: "/configuraciones", texto: "Configuraciones", Icono: FileCode2 },
  { to: "/trabajos", texto: "Trabajos", Icono: ListChecks },
  { to: "/resultados", texto: "Resultados", Icono: Images },
  { to: "/ajustes", texto: "Ajustes", Icono: Settings },
] as const;

function ServidorCaido() {
  const { errorEstado, recomprobar, baseUrl } = useAtalaya();
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-lg rounded-lg border border-destructive/40 bg-card p-6">
        <h1 className="text-xl font-semibold">Servidor no disponible</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          No se pudo conectar con la API de ATALAYA en{" "}
          <span className="font-mono text-foreground">{baseUrl}</span>
          {errorEstado ? ` (${errorEstado})` : ""}. Arráncalo con:
        </p>
        <pre className="mt-3 rounded-md bg-terminal p-3 font-mono text-sm text-terminal-foreground">
          cd atalaya_fw && python servidor_api.py
        </pre>
        <div className="mt-4 flex gap-2">
          <Button onClick={recomprobar}>Reintentar</Button>
          <Button variant="outline" asChild>
            <Link to="/ajustes">Cambiar URL de la API</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const { estado, cargandoEstado, configs, configActiva, setConfigActiva } =
    useAtalaya();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  if (cargandoEstado && !estado) {
    return (
      <div className="flex min-h-screen items-center justify-center font-mono text-sm text-muted-foreground">
        conectando con el servidor…
      </div>
    );
  }
  if (!estado) return <ServidorCaido />;

  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar md:flex">
        <div className="flex items-center gap-2 px-4 py-4">
          <Radar className="size-5 text-primary" />
          <span className="font-mono text-lg font-semibold tracking-widest text-primary">
            ATALAYA
          </span>
        </div>
        <nav className="flex flex-col gap-1 px-2">
          {enlaces.map(({ to, texto, Icono }) => (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-2 rounded-md px-3 py-2 text-sm text-sidebar-foreground transition-colors hover:bg-sidebar-accent",
                pathname === to && "bg-sidebar-accent text-primary",
              )}
            >
              <Icono className="size-4" />
              {texto}
            </Link>
          ))}
        </nav>
        <p className="mt-auto px-4 py-4 font-mono text-[11px] text-muted-foreground">
          odometría visual-inercial
        </p>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center gap-3 border-b border-border bg-card px-4 py-2">
          <span className="flex items-center gap-2 font-mono text-xs">
            <span className="size-2 rounded-full bg-exito" />
            conectado · v{estado.version}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <span className="font-mono text-xs text-muted-foreground">config</span>
            <Select value={configActiva ?? ""} onValueChange={setConfigActiva}>
              <SelectTrigger className="h-8 w-64 font-mono text-xs">
                <SelectValue placeholder="sin configuraciones" />
              </SelectTrigger>
              <SelectContent>
                {configs.map((c) => (
                  <SelectItem key={c} value={c} className="font-mono text-xs">
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </header>

        {!estado.pymavlink && (
          <div className="flex items-start gap-2 border-b border-aviso/40 bg-aviso/10 px-4 py-2 text-sm text-aviso">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span>
              pymavlink no está instalado: no se podrán leer logs de ArduPilot (
              <span className="font-mono">pip install pymavlink</span>).
            </span>
          </div>
        )}

        <main className="min-w-0 flex-1 p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}
