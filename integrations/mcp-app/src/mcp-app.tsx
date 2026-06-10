import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

interface IdealizeLogoResult {
  image_path: string;
  output_path: string | null;
  width: number;
  height: number;
  svg_bytes: number;
  svg: string;
}

interface IdealizeLogoInput {
  image_path?: string;
  output_path?: string | null;
  epsilon?: number;
  max_error?: number;
  colors?: number;
  flatten?: boolean;
  no_symmetry?: boolean;
}

function firstText(result: CallToolResult): string | undefined {
  const block = result.content?.find((item) => item.type === "text");
  return block?.type === "text" ? block.text : undefined;
}

function parseStructuredResult(result: CallToolResult | null): IdealizeLogoResult | null {
  if (!result) return null;
  const structured = result.structuredContent as Partial<IdealizeLogoResult> | undefined;
  if (structured?.svg && structured.image_path) {
    return {
      image_path: structured.image_path,
      output_path: structured.output_path ?? null,
      width: Number(structured.width ?? 0),
      height: Number(structured.height ?? 0),
      svg_bytes: Number(structured.svg_bytes ?? structured.svg.length),
      svg: structured.svg,
    };
  }

  const text = firstText(result);
  if (!text) return null;
  try {
    const parsed = JSON.parse(text) as Partial<IdealizeLogoResult>;
    if (parsed.svg && parsed.image_path) {
      return {
        image_path: parsed.image_path,
        output_path: parsed.output_path ?? null,
        width: Number(parsed.width ?? 0),
        height: Number(parsed.height ?? 0),
        svg_bytes: Number(parsed.svg_bytes ?? parsed.svg.length),
        svg: parsed.svg,
      };
    }
  } catch {
    return null;
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function AppRoot() {
  const [toolInput, setToolInput] = useState<IdealizeLogoInput>({});
  const [toolResult, setToolResult] = useState<CallToolResult | null>(null);
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [errorText, setErrorText] = useState<string | null>(null);

  const { app, isConnected, error } = useApp({
    appInfo: { name: "vectormark", version: "0.0.1" },
    capabilities: {},
    onAppCreated: (createdApp) => {
      createdApp.ontoolinput = (params) => {
        setToolInput((current) => ({
          ...current,
          ...(params.arguments as IdealizeLogoInput),
        }));
      };
      createdApp.ontoolresult = (result) => {
        setErrorText(null);
        setToolResult(result);
      };
      createdApp.ontoolcancelled = (params) => {
        setErrorText(params.reason || "Tool call was cancelled.");
      };
      createdApp.onerror = (appError) => {
        setErrorText(appError instanceof Error ? appError.message : String(appError));
      };
      createdApp.onhostcontextchanged = (params) => {
        setHostContext((current) => ({ ...current, ...params }));
      };
      createdApp.onteardown = async () => ({});
    },
  });

  useEffect(() => {
    if (app) setHostContext(app.getHostContext());
  }, [app]);

  const result = useMemo(() => parseStructuredResult(toolResult), [toolResult]);

  if (error) {
    return <StatusPanel title="Connection failed" detail={error.message} />;
  }

  if (!isConnected || !app) {
    return <StatusPanel title="Connecting" detail="Waiting for the MCP Apps host." />;
  }

  return (
    <VectormarkApp
      app={app}
      hostContext={hostContext}
      initialInput={toolInput}
      result={result}
      errorText={errorText}
      onResult={setToolResult}
      onError={setErrorText}
    />
  );
}

interface VectormarkAppProps {
  app: App;
  hostContext?: McpUiHostContext;
  initialInput: IdealizeLogoInput;
  result: IdealizeLogoResult | null;
  errorText: string | null;
  onResult: (result: CallToolResult | null) => void;
  onError: (message: string | null) => void;
}

function VectormarkApp({
  app,
  hostContext,
  initialInput,
  result,
  errorText,
  onResult,
  onError,
}: VectormarkAppProps) {
  const [form, setForm] = useState<IdealizeLogoInput>({
    colors: 16,
    epsilon: 1.5,
    max_error: 1,
    flatten: false,
    no_symmetry: false,
  });
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    setForm((current) => ({ ...current, ...initialInput }));
  }, [initialInput]);

  const runIdealize = useCallback(async () => {
    if (!form.image_path?.trim()) {
      onError("Choose a local raster path before running vectormark.");
      return;
    }

    setIsRunning(true);
    onError(null);
    try {
      const nextResult = await app.callServerTool({
        name: "idealize_logo",
        arguments: {
          image_path: form.image_path,
          output_path: form.output_path || undefined,
          colors: Number(form.colors ?? 16),
          epsilon: Number(form.epsilon ?? 1.5),
          max_error: Number(form.max_error ?? 1),
          flatten: Boolean(form.flatten),
          no_symmetry: Boolean(form.no_symmetry),
        },
      });
      onResult(nextResult);
    } catch (runError) {
      onError(runError instanceof Error ? runError.message : String(runError));
    } finally {
      setIsRunning(false);
    }
  }, [app, form, onError, onResult]);

  return (
    <main
      className="app-shell"
      style={{
        paddingTop: hostContext?.safeAreaInsets?.top,
        paddingRight: hostContext?.safeAreaInsets?.right,
        paddingBottom: hostContext?.safeAreaInsets?.bottom,
        paddingLeft: hostContext?.safeAreaInsets?.left,
      }}
    >
      <section className="control-panel" aria-label="vectormark controls">
        <div className="brand-strip">
          <div>
            <p className="eyebrow">vectormark</p>
            <h1>Logo idealizer</h1>
          </div>
          <span className="status-dot" aria-label={isRunning ? "Running" : "Ready"} />
        </div>

        <label className="field field-wide">
          Raster path
          <input
            value={form.image_path ?? ""}
            onChange={(event) => setForm((current) => ({ ...current, image_path: event.target.value }))}
            placeholder="/path/to/logo.png"
          />
        </label>

        <label className="field field-wide">
          SVG output path
          <input
            value={form.output_path ?? ""}
            onChange={(event) => setForm((current) => ({ ...current, output_path: event.target.value }))}
            placeholder="Optional"
          />
        </label>

        <div className="number-grid">
          <label className="field">
            Colors
            <input
              min="1"
              max="64"
              type="number"
              value={form.colors ?? 16}
              onChange={(event) => setForm((current) => ({ ...current, colors: Number(event.target.value) }))}
            />
          </label>
          <label className="field">
            Epsilon
            <input
              min="0"
              step="0.1"
              type="number"
              value={form.epsilon ?? 1.5}
              onChange={(event) => setForm((current) => ({ ...current, epsilon: Number(event.target.value) }))}
            />
          </label>
          <label className="field">
            Max error
            <input
              min="0"
              step="0.1"
              type="number"
              value={form.max_error ?? 1}
              onChange={(event) => setForm((current) => ({ ...current, max_error: Number(event.target.value) }))}
            />
          </label>
        </div>

        <div className="toggle-row">
          <label className="toggle">
            <input
              checked={Boolean(form.flatten)}
              type="checkbox"
              onChange={(event) => setForm((current) => ({ ...current, flatten: event.target.checked }))}
            />
            Flatten paths
          </label>
          <label className="toggle">
            <input
              checked={Boolean(form.no_symmetry)}
              type="checkbox"
              onChange={(event) => setForm((current) => ({ ...current, no_symmetry: event.target.checked }))}
            />
            No symmetry
          </label>
        </div>

        <button className="run-button" disabled={isRunning} onClick={runIdealize}>
          {isRunning ? "Idealizing..." : "Run vectormark"}
        </button>
      </section>

      <section className="preview-panel" aria-label="SVG preview">
        {errorText ? <div className="error-banner">{errorText}</div> : null}
        {result ? <LogoPreview result={result} /> : <EmptyPreview />}
      </section>
    </main>
  );
}

function LogoPreview({ result }: { result: IdealizeLogoResult }) {
  // Render via <img> data URI, NOT dangerouslySetInnerHTML: an <img>-loaded SVG cannot
  // execute scripts or event handlers, so a hostile `svg` (e.g. from render_idealized_logo,
  // which echoes caller-supplied SVG) can't run in the widget. encodeURIComponent keeps it
  // UTF-8 safe (btoa would choke on non-Latin1).
  const svgSrc = `data:image/svg+xml,${encodeURIComponent(result.svg)}`;
  return (
    <>
      <div className="preview-stage">
        <img src={svgSrc} alt="Idealized logo preview" />
      </div>
      <dl className="metrics">
        <div>
          <dt>Canvas</dt>
          <dd>{result.width} x {result.height}</dd>
        </div>
        <div>
          <dt>SVG</dt>
          <dd>{formatBytes(result.svg_bytes)}</dd>
        </div>
        <div>
          <dt>Output</dt>
          <dd>{result.output_path || "inline result"}</dd>
        </div>
      </dl>
    </>
  );
}

function EmptyPreview() {
  return (
    <div className="empty-preview">
      <div className="empty-glyph" />
      <p>Run vectormark to preview the idealized SVG.</p>
    </div>
  );
}

function StatusPanel({ title, detail }: { title: string; detail: string }) {
  return (
    <main className="status-panel">
      <h1>{title}</h1>
      <p>{detail}</p>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppRoot />
  </StrictMode>,
);
