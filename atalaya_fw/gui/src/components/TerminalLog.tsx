import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";

export function TerminalLog({
  texto,
  altura = "h-64",
  titulo,
}: {
  texto: string;
  altura?: string;
  titulo?: string;
}) {
  const [autoscroll, setAutoscroll] = useState(true);
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (autoscroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [texto, autoscroll]);

  return (
    <div className="overflow-hidden rounded-md border border-border">
      <div className="flex items-center justify-between border-b border-border bg-card px-3 py-1.5">
        <span className="font-mono text-xs text-muted-foreground">
          {titulo ?? "log"}
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 font-mono text-xs"
          onClick={() => setAutoscroll((a) => !a)}
        >
          {autoscroll ? "autoscroll: on" : "autoscroll: fijo"}
        </Button>
      </div>
      <pre
        ref={ref}
        className={`${altura} overflow-auto bg-terminal p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-terminal-foreground`}
      >
        {texto || "— sin salida todavía —"}
      </pre>
    </div>
  );
}
