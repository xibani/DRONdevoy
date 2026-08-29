import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { useAtalaya } from "@/lib/atalaya-store";
import { api, duracion, type Comando, type Trabajo } from "@/lib/api";
import { TerminalLog } from "@/components/TerminalLog";
import { BadgeEstado } from "@/components/BadgeEstado";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { AlertTriangle, CheckCircle2, ChevronDown, XCircle } from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Flujo de trabajo — ATALAYA VIO" },
      {
        name: "description",
        content:
          "Asistente de 5 pasos para procesar vuelos de dron con odometría visual-inercial: selftest, inspección de log, extracción de frames, offset temporal y ejecución del pipeline.",
      },
      { property: "og:title", content: "Flujo de trabajo — ATALAYA VIO" },
      {
        property: "og:description",
        content:
          "Panel de control del pipeline VIO: selftest, inspección, frames, offset y ejecución.",
      },
    ],
  }),
  component: Flujo,
});

function Paso({
  numero,
  titulo,
  ayuda,
  trabajo,
  children,
}: {
  numero: number;
  titulo: string;
  ayuda: string;
  trabajo?: Trabajo;
  children: React.ReactNode;
}) {
  const { logs, seguirLog } = useAtalaya();
  const [abierto, setAbierto] = useState(false);
  const log = trabajo ? logs[trabajo.id]?.texto : undefined;

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-md border border-primary/40 bg-primary/10 font-mono text-sm text-primary">
          {numero}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-semibold">{titulo}</h2>
            {trabajo && <BadgeEstado estado={trabajo.estado} />}
            {trabajo && (
              <span className="font-mono text-xs text-muted-foreground">
                {duracion(trabajo)}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{ayuda}</p>
          <div className="mt-3 space-y-3">{children}</div>
          {trabajo && (
            <div className="mt-3">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 font-mono text-xs"
                onClick={() => {
                  seguirLog(trabajo.id);
                  setAbierto((a) => !a);
                }}
              >
                <ChevronDown
                  className={`size-3.5 transition-transform ${abierto ? "rotate-180" : ""}`}
                />
                log en vivo
              </Button>
              {abierto && (
                <div className="mt-2">
                  <TerminalLog texto={log ?? ""} titulo={trabajo.id.slice(0, 8)} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function BotonLanzar({
  comando,
  texto,
  extra,
  onLanzado,
}: {
  comando: Comando;
  texto: string;
  extra?: Record<string, unknown>;
  onLanzado?: (id: string) => void;
}) {
  const { lanzar, ocupado, configActiva } = useAtalaya();
  const enCurso = ocupado(comando);
  const faltaConfig = comando !== "selftest" && !configActiva;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        disabled={enCurso || faltaConfig}
        onClick={async () => {
          try {
            const id = await lanzar(comando, extra ?? {});
            onLanzado?.(id);
          } catch (e) {
            toast.error((e as Error).message);
          }
        }}
      >
        {texto}
      </Button>
      {enCurso && (
        <span className="font-mono text-xs text-muted-foreground">
          ya hay un trabajo de este tipo en curso
        </span>
      )}
      {faltaConfig && (
        <span className="font-mono text-xs text-muted-foreground">
          selecciona una configuración activa
        </span>
      )}
    </div>
  );
}

function num(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}

function Flujo() {
  const { ultimo, baseUrl, configActiva } = useAtalaya();
  const [tIni, setTIni] = useState("");
  const [tFin, setTFin] = useState("");
  const [rango, setRango] = useState("2.0");
  const [sinDR, setSinDR] = useState(false);
  const [aplicando, setAplicando] = useState(false);

  const tSelftest = ultimo("selftest");
  const tInspec = ultimo("inspeccionar");
  const tFrames = ultimo("extraer-frames");
  const tOffset = ultimo("offset");
  const tEjec = ultimo("ejecutar");

  const resOffset = (tOffset?.resultado ?? {}) as Record<string, unknown>;
  const offsetS = tOffset?.estado === "ok" ? num(resOffset["offset_s"]) : undefined;
  const avisoContraste = resOffset["aviso_contraste"] === true;

  const resEjec = (tEjec?.resultado ?? {}) as Record<string, unknown>;
  const aceptadas = num(resEjec["aceptadas_pct"]);
  const directorio =
    typeof resEjec["directorio"] === "string" ? (resEjec["directorio"] as string) : null;
  // el servidor imprime la ruta con separadores del SO y barra final
  const nombreInforme =
    directorio
      ?.replace(/[\\/]+$/, "")
      .split(/[\\/]/)
      .pop() ?? null;

  const semaforo =
    aceptadas === undefined
      ? "text-muted-foreground"
      : aceptadas >= 80
        ? "text-exito"
        : aceptadas >= 50
          ? "text-aviso"
          : "text-destructive";

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Flujo de trabajo</h1>
        <p className="text-sm text-muted-foreground">
          Cinco pasos sobre la configuración activa{" "}
          <span className="font-mono text-foreground">{configActiva ?? "—"}</span>. Los
          pasos no se bloquean: puedes saltar entre ellos.
        </p>
      </div>

      <Paso
        numero={0}
        titulo="Selftest"
        ayuda="Valida la matemática sin datos; si falla, el problema es del entorno Python, no de tus datos."
        {...(tSelftest ? { trabajo: tSelftest } : {})}
      >
        <BotonLanzar comando="selftest" texto="Ejecutar selftest" />
        {tSelftest?.estado === "ok" && (
          <p className="flex items-center gap-2 text-sm text-exito">
            <CheckCircle2 className="size-4" /> TODO OK
          </p>
        )}
        {tSelftest?.estado === "error" && (
          <p className="flex items-center gap-2 text-sm text-destructive">
            <XCircle className="size-4" /> falló ·{" "}
            <Link
              to="/trabajos"
              search={{ id: tSelftest.id }}
              className="underline underline-offset-2"
            >
              ver log
            </Link>
          </p>
        )}
      </Paso>

      <Paso
        numero={1}
        titulo="Inspeccionar el log"
        ayuda="Con esto decides dataset.imu.mensaje (IMU o GYR+ACC), instancia y gt.fuente en la configuración."
        {...(tInspec ? { trabajo: tInspec } : {})}
      >
        <BotonLanzar comando="inspeccionar" texto="Inspeccionar log" />
      </Paso>

      <Paso
        numero={2}
        titulo="Extraer frames"
        ayuda="Solo hace falta una vez por vídeo; si los frames ya existen, se reutilizan."
        {...(tFrames ? { trabajo: tFrames } : {})}
      >
        <BotonLanzar comando="extraer-frames" texto="Extraer frames" />
      </Paso>

      <Paso
        numero={3}
        titulo="Offset temporal"
        ayuda="Elige un tramo del vuelo con giros claros."
        {...(tOffset ? { trabajo: tOffset } : {})}
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div>
            <Label className="font-mono text-xs">t_ini (s)</Label>
            <Input
              className="mt-1 font-mono"
              value={tIni}
              onChange={(e) => setTIni(e.target.value)}
              placeholder="20"
            />
          </div>
          <div>
            <Label className="font-mono text-xs">t_fin (s)</Label>
            <Input
              className="mt-1 font-mono"
              value={tFin}
              onChange={(e) => setTFin(e.target.value)}
              placeholder="60"
            />
          </div>
          <div>
            <Label className="font-mono text-xs">rango (s)</Label>
            <Input
              className="mt-1 font-mono"
              value={rango}
              onChange={(e) => setRango(e.target.value)}
            />
          </div>
        </div>
        <BotonLanzar
          comando="offset"
          texto="Estimar"
          extra={{
            ...(tIni !== "" ? { t_ini: Number(tIni) } : {}),
            ...(tFin !== "" ? { t_fin: Number(tFin) } : {}),
            ...(rango !== "" ? { rango: Number(rango) } : {}),
          }}
        />
        {offsetS !== undefined && (
          <div className="rounded-md border border-border bg-background p-4">
            <p className="font-mono text-xs text-muted-foreground">offset estimado</p>
            <p className="font-mono text-4xl font-semibold text-primary">
              {(offsetS * 1000).toFixed(1)} ms
            </p>
            <p className="font-mono text-xs text-muted-foreground">
              ({offsetS.toFixed(4)} s)
            </p>
            {avisoContraste && (
              <p className="mt-3 flex items-start gap-2 rounded border border-aviso/40 bg-aviso/10 p-2 text-sm text-aviso">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                Poca rotación en el tramo elegido; el offset no es fiable, elige otro
                intervalo con giros.
              </p>
            )}
            <Button
              className="mt-3"
              variant="outline"
              disabled={aplicando || !configActiva}
              onClick={async () => {
                if (!configActiva) return;
                setAplicando(true);
                try {
                  await api.campoConfig(
                    baseUrl,
                    configActiva,
                    "dataset.camara.offset_temporal_s",
                    offsetS,
                  );
                  toast.success("Offset aplicado a la configuración");
                } catch (e) {
                  toast.error((e as Error).message);
                } finally {
                  setAplicando(false);
                }
              }}
            >
              Aplicar a la configuración
            </Button>
            <p className="mt-2 text-xs text-muted-foreground">
              Aviso: al aplicar se reescribe el YAML y se pierden los comentarios.
            </p>
          </div>
        )}
      </Paso>

      <Paso
        numero={4}
        titulo="Ejecutar pipeline"
        ayuda="Corre el pipeline completo y genera el informe con gráficas y métricas."
        {...(tEjec ? { trabajo: tEjec } : {})}
      >
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={sinDR}
            onCheckedChange={(v) => setSinDR(v === true)}
          />
          sin dead reckoning (más rápido)
        </label>
        <BotonLanzar
          comando="ejecutar"
          texto="Ejecutar pipeline"
          extra={{ sin_dead_reckoning: sinDR }}
        />
        {tEjec?.estado === "ok" && (
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {[
                { k: "ate_se3", etiqueta: "ATE SE3" },
                { k: "ate_sim3", etiqueta: "ATE Sim3" },
              ].map(({ k, etiqueta }) => (
                <div key={k} className="rounded-md border border-border p-3">
                  <p className="font-mono text-xs text-muted-foreground">{etiqueta}</p>
                  <p className="font-mono text-2xl">
                    {num(resEjec[k])?.toFixed(3) ?? "—"}
                  </p>
                </div>
              ))}
              <div className="rounded-md border border-border p-3">
                <p className="font-mono text-xs text-muted-foreground">aceptadas</p>
                <p className={`font-mono text-2xl ${semaforo}`}>
                  {aceptadas !== undefined ? `${aceptadas.toFixed(1)} %` : "—"}
                </p>
              </div>
            </div>
            {aceptadas !== undefined && aceptadas < 50 && (
              <p className="rounded border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive">
                Revisa offset temporal, extrínseca T_body_cam o unidades de la IMU.
              </p>
            )}
            {nombreInforme && (
              <Button variant="outline" asChild>
                <Link to="/resultados" search={{ nombre: nombreInforme }}>
                  Ver resultados de {nombreInforme}
                </Link>
              </Button>
            )}
          </div>
        )}
      </Paso>
    </div>
  );
}
