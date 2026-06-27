import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

// ─── Data shapes ──────────────────────────────────────────────────────────────

interface DiagnosticsInput {
  source_kind?: string;
  mime_type?: string;
  bytes?: number;
  sha256?: string;
  original_width?: number;
  original_height?: number;
}

interface DiagnosticsProcessed {
  width?: number;
  height?: number;
  cropped?: boolean;
  resized?: boolean;
  transparent?: boolean;
  quantized?: boolean;
}

interface DiagnosticsVectormark {
  colors?: number;
  flatten?: boolean;
  no_symmetry?: boolean;
  epsilon?: number;
  max_error?: number;
}

interface DiagnosticsOutput {
  svg_bytes?: number;
  element_count?: number;
  has_defs?: boolean;
  has_paths?: boolean;
  has_primitives?: boolean;
  has_symmetry?: boolean;
}

interface Diagnostics {
  input?: DiagnosticsInput;
  processed?: DiagnosticsProcessed;
  vectormark?: DiagnosticsVectormark;
  output?: DiagnosticsOutput;
  warnings?: string[];
}

/** Richer idealize_logo shape; also accepts the older flat render_idealized_logo shape. */
interface IdealizeLogoResult {
  svg: string;
  width: number;
  height: number;
  svg_bytes: number;
  preview_available?: boolean;
  /** Legacy fields — accepted for backward compat but never displayed. */
  image_path?: string;
  output_path?: string | null;
  diagnostics?: Diagnostics;
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

type Theme = "default" | "daikonic";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function firstText(result: CallToolResult): string | undefined {
  const block = result.content?.find((item) => item.type === "text");
  return block?.type === "text" ? block.text : undefined;
}

function firstImageDataUri(result: CallToolResult | null): string | null {
  if (!result) return null;
  const block = result.content?.find((item) => item.type === "image");
  if (block?.type === "image" && block.data && block.mimeType) {
    return `data:${block.mimeType};base64,${block.data}`;
  }
  return null;
}

/**
 * Accepts both the richer idealize_logo shape (with optional diagnostics) and
 * the older flat render_idealized_logo shape. Defensive: missing fields are
 * tolerated — only `svg` is required.
 */
function parseStructuredResult(result: CallToolResult | null): IdealizeLogoResult | null {
  if (!result) return null;

  const structured = result.structuredContent as Partial<IdealizeLogoResult> | undefined;
  if (structured?.svg) {
    return {
      svg: structured.svg,
      width: Number(structured.width ?? 0),
      height: Number(structured.height ?? 0),
      svg_bytes: Number(structured.svg_bytes ?? structured.svg.length),
      preview_available: structured.preview_available ?? false,
      image_path: structured.image_path,
      output_path: structured.output_path ?? null,
      diagnostics: structured.diagnostics,
    };
  }

  const text = firstText(result);
  if (!text) return null;
  try {
    const parsed = JSON.parse(text) as Partial<IdealizeLogoResult>;
    if (parsed.svg) {
      return {
        svg: parsed.svg,
        width: Number(parsed.width ?? 0),
        height: Number(parsed.height ?? 0),
        svg_bytes: Number(parsed.svg_bytes ?? parsed.svg.length),
        preview_available: parsed.preview_available ?? false,
        image_path: parsed.image_path,
        output_path: parsed.output_path ?? null,
        diagnostics: parsed.diagnostics,
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

const SOURCE_KIND_LABELS: Record<string, string> = {
  platform_file: "Platform file",
  local_path: "Local file",
  url: "URL",
  data_uri: "Data URI",
  base64: "Base64",
};

// ─── Daikonic brand mark (trusted static SVG, rendered via JSX) ───────────────

function DaikonicMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 1254 1254"
      width={size}
      height={size}
      aria-hidden="true"
      style={{ flexShrink: 0, display: "block" }}
    >
      <path
        fill="#031D31"
        d="M622.5 360.5 L625.1 360.2 C625.87 360.09 626.47 359.37 626.43 358.59 Q625.75 343.98 627.5 329 Q628.45 250.49 670.5 189 Q679.51 178 688.5 167 Q740.36 114.5 809 102.5 Q821.75 101.39 835 101.5 Q858.06 103.11 872.5 122 Q876 129.79 878.5 138 Q882.17 172.19 870.5 202 Q860.07 226.77 843.5 249 Q791.62 308.08 721 325.5 Q701.08 332.4 680 336.5 Q678 337.5 676 338.5 Q655.27 341.77 639.5 357 Q638.61 358.84 638.58 359.35 C638.54 360.26 639.24 361.02 640.15 361.03 Q666.64 361.59 693 366.5 Q798.97 379.91 876.5 462 Q885.86 474.06 895.5 486 Q913.75 513.15 924.8 541.23 C925.19 542.21 925.23 543.81 924.9 544.8 L924.6 545.7 C924.27 546.69 923.15 547.5 922.1 547.5 L622.5 547.5 L322.9 547.5 C321.85 547.5 320.73 546.69 320.4 545.7 L320.1 544.8 C319.77 543.81 319.81 542.21 320.2 541.23 Q331.25 513.15 349.5 486 Q359.14 474.06 368.5 462 Q446.03 379.91 552 366.5 Q578.36 361.59 604.85 361.03 C605.76 361.02 606.46 360.26 606.42 359.35 Q606.39 358.84 605.5 357 Q589.73 341.77 569 338.5 Q567 337.5 565 336.5 Q543.92 332.4 524 325.5 Q453.38 308.08 401.5 249 Q384.93 226.77 374.5 202 Q362.83 172.19 366.5 138 Q369 129.79 372.5 122 Q386.94 103.11 410 101.5 Q423.25 101.39 436 102.5 Q504.64 114.5 556.5 167 Q565.49 178 574.5 189 Q616.55 250.49 617.5 329 Q619.25 343.98 618.57 358.59 C618.53 359.37 619.13 360.09 619.9 360.2 L622.5 360.5 Z"
      />
      <path
        fill="#1FAB9F"
        d="M622.5 578.5 L928.8 578.5 C931.12 578.5 934.2 579.95 935.69 581.73 Q936.51 582.73 937.5 588 Q939.82 606.08 942.5 624 Q943.77 667.77 936.5 708 Q935.31 710.31 934.76 710.6 C933.79 711.1 932.11 711.5 931.02 711.5 L622.5 711.5 L313.98 711.5 C312.89 711.5 311.21 711.1 310.24 710.6 Q309.69 710.31 308.5 708 Q301.23 667.77 302.5 624 Q305.18 606.08 307.5 588 Q308.49 582.73 309.31 581.73 C310.8 579.95 313.88 578.5 316.2 578.5 L622.5 578.5 Z"
      />
      <path
        fill="#FE830E"
        d="M622.5 743.5 L920.9 744.96 C925.1 744.98 927.74 748.32 926.8 752.41 Q913.08 812.26 879.5 872 Q877.75 873.75 877.4 874.1 C876.63 874.87 875.11 875.5 874.02 875.5 L622.5 875.5 L370.98 875.5 C369.89 875.5 368.37 874.87 367.6 874.1 Q367.25 873.75 365.5 872 Q331.92 812.26 318.2 752.41 C317.26 748.32 319.9 744.98 324.1 744.96 L622.5 743.5 Z"
      />
      <path
        fill="#ED3125"
        d="M622.5 907.5 L848.9 907.98 C853.1 907.99 854.51 910.76 852.05 914.16 Q846.69 921.58 836.5 935 Q786 991.62 723 1021.5 Q707.96 1030.4 693 1039.5 Q675.27 1052.08 660.5 1069 Q638.41 1098.85 631.5 1135 Q629.99 1137.85 629.37 1138.68 C628.61 1139.68 627.02 1140.77 625.8 1141.1 L622.5 1142 L619.2 1141.1 C617.98 1140.77 616.39 1139.68 615.63 1138.68 Q615.01 1137.85 613.5 1135 Q606.59 1098.85 584.5 1069 Q569.73 1052.08 552 1039.5 Q537.04 1030.4 522 1021.5 Q459 991.62 408.5 935 Q398.31 921.58 392.95 914.16 C390.49 910.76 391.9 907.99 396.1 907.98 L622.5 907.5 Z"
      />
    </svg>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────

function AppRoot() {
  const [toolInput, setToolInput] = useState<IdealizeLogoInput>({});
  const [toolResult, setToolResult] = useState<CallToolResult | null>(null);
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [errorText, setErrorText] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>("default");

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

  // Apply daikonic theme to the document root so CSS variables cascade everywhere.
  useEffect(() => {
    document.documentElement.dataset.theme = theme === "daikonic" ? "daikonic" : "";
    return () => {
      delete document.documentElement.dataset.theme;
    };
  }, [theme]);

  const result = useMemo(() => parseStructuredResult(toolResult), [toolResult]);
  const rasterSrc = useMemo(() => firstImageDataUri(toolResult), [toolResult]);

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
      rasterSrc={rasterSrc}
      errorText={errorText}
      theme={theme}
      onThemeToggle={() => setTheme((t) => (t === "daikonic" ? "default" : "daikonic"))}
      onResult={setToolResult}
      onError={setErrorText}
    />
  );
}

// ─── VectormarkApp ────────────────────────────────────────────────────────────

interface VectormarkAppProps {
  app: App;
  hostContext?: McpUiHostContext;
  initialInput: IdealizeLogoInput;
  result: IdealizeLogoResult | null;
  rasterSrc: string | null;
  errorText: string | null;
  theme: Theme;
  onThemeToggle: () => void;
  onResult: (result: CallToolResult | null) => void;
  onError: (message: string | null) => void;
}

function VectormarkApp({
  app,
  hostContext,
  initialInput,
  result,
  rasterSrc,
  errorText,
  theme,
  onThemeToggle,
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
          <div className="brand-id">
            {theme === "daikonic" && <DaikonicMark size={32} />}
            <div>
              <p className="eyebrow">vectormark</p>
              <h1>Logo idealizer</h1>
            </div>
          </div>
          <div className="brand-strip-end">
            <button
              className="theme-btn"
              title={
                theme === "daikonic"
                  ? "Switch to default theme"
                  : "Switch to daikonic brand theme"
              }
              onClick={onThemeToggle}
            >
              {theme === "daikonic" ? "Default" : "Daikonic"}
            </button>
            <span className="status-dot" aria-label={isRunning ? "Running" : "Ready"} />
          </div>
        </div>

        <label className="field field-wide">
          Raster path
          <input
            value={form.image_path ?? ""}
            onChange={(event) =>
              setForm((current) => ({ ...current, image_path: event.target.value }))
            }
            placeholder="/path/to/logo.png"
          />
        </label>

        <label className="field field-wide">
          SVG output path
          <input
            value={form.output_path ?? ""}
            onChange={(event) =>
              setForm((current) => ({ ...current, output_path: event.target.value }))
            }
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
              onChange={(event) =>
                setForm((current) => ({ ...current, colors: Number(event.target.value) }))
              }
            />
          </label>
          <label className="field">
            Epsilon
            <input
              min="0"
              step="0.1"
              type="number"
              value={form.epsilon ?? 1.5}
              onChange={(event) =>
                setForm((current) => ({ ...current, epsilon: Number(event.target.value) }))
              }
            />
          </label>
          <label className="field">
            Max error
            <input
              min="0"
              step="0.1"
              type="number"
              value={form.max_error ?? 1}
              onChange={(event) =>
                setForm((current) => ({ ...current, max_error: Number(event.target.value) }))
              }
            />
          </label>
        </div>

        <div className="toggle-row">
          <label className="toggle">
            <input
              checked={Boolean(form.flatten)}
              type="checkbox"
              onChange={(event) =>
                setForm((current) => ({ ...current, flatten: event.target.checked }))
              }
            />
            Flatten paths
          </label>
          <label className="toggle">
            <input
              checked={Boolean(form.no_symmetry)}
              type="checkbox"
              onChange={(event) =>
                setForm((current) => ({ ...current, no_symmetry: event.target.checked }))
              }
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
        {result ? <LogoPreview result={result} rasterSrc={rasterSrc} /> : <EmptyPreview />}
      </section>
    </main>
  );
}

// ─── LogoPreview ──────────────────────────────────────────────────────────────

function LogoPreview({
  result,
  rasterSrc,
}: {
  result: IdealizeLogoResult;
  rasterSrc: string | null;
}) {
  const [view, setView] = useState<"svg" | "raster">("svg");
  const [copied, setCopied] = useState(false);

  // Render via <img> data URI, NOT dangerouslySetInnerHTML: an <img>-loaded SVG cannot
  // execute scripts or event handlers, so a hostile `svg` (e.g. from render_idealized_logo,
  // which echoes caller-supplied SVG) can't run in the widget. encodeURIComponent keeps it
  // UTF-8 safe (btoa would choke on non-Latin1).
  const svgSrc = `data:image/svg+xml,${encodeURIComponent(result.svg)}`;
  const showRaster = view === "raster" && rasterSrc != null;
  const hasRaster = rasterSrc != null;

  const copySvg = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(result.svg);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access denied — fail silently.
    }
  }, [result.svg]);

  const downloadSvg = useCallback(() => {
    const blob = new Blob([result.svg], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "logo.svg";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [result.svg]);

  return (
    <>
      {hasRaster && (
        <div className="view-toggle" role="group" aria-label="Preview mode">
          <button
            className={`view-btn${view === "raster" ? " active" : ""}`}
            onClick={() => setView("raster")}
          >
            Before
          </button>
          <button
            className={`view-btn${view === "svg" ? " active" : ""}`}
            onClick={() => setView("svg")}
          >
            After
          </button>
        </div>
      )}

      <div className="preview-stage">
        <img
          src={showRaster ? rasterSrc : svgSrc}
          alt={showRaster ? "Original raster input" : "Idealized logo SVG"}
        />
      </div>

      <div className="preview-footer">
        <dl className="metrics">
          <div>
            <dt>Canvas</dt>
            <dd>
              {result.width} × {result.height}
            </dd>
          </div>
          <div>
            <dt>SVG</dt>
            <dd>{formatBytes(result.svg_bytes)}</dd>
          </div>
        </dl>
        <div className="action-buttons">
          <button className="action-btn" onClick={copySvg}>
            {copied ? "Copied!" : "Copy SVG"}
          </button>
          <button className="action-btn" onClick={downloadSvg}>
            Download
          </button>
        </div>
      </div>

      {result.diagnostics != null && (
        <DiagnosticsPanel diagnostics={result.diagnostics} />
      )}
    </>
  );
}

// ─── DiagnosticsPanel ─────────────────────────────────────────────────────────

function DiagnosticsPanel({ diagnostics }: { diagnostics: Diagnostics }) {
  const { input, processed, vectormark, output, warnings } = diagnostics;
  const warnCount = warnings?.length ?? 0;

  return (
    <details className="diag-panel">
      <summary className="diag-summary">
        Diagnostics
        {warnCount > 0 && (
          <span className="diag-warn-badge" title={`${warnCount} warning${warnCount > 1 ? "s" : ""}`}>
            {warnCount}
          </span>
        )}
      </summary>

      <div className="diag-body">
        {input != null && (
          <div className="diag-section">
            <span className="diag-section-title">Input</span>
            <div className="diag-rows">
              {input.source_kind != null && (
                <div className="diag-row">
                  <span className="diag-label">Source</span>
                  <span className="diag-value">
                    {SOURCE_KIND_LABELS[input.source_kind] ?? input.source_kind}
                  </span>
                </div>
              )}
              {input.mime_type != null && (
                <div className="diag-row">
                  <span className="diag-label">Type</span>
                  <span className="diag-value">{input.mime_type}</span>
                </div>
              )}
              {input.original_width != null && input.original_height != null && (
                <div className="diag-row">
                  <span className="diag-label">Original</span>
                  <span className="diag-value">
                    {input.original_width} × {input.original_height}
                  </span>
                </div>
              )}
              {input.bytes != null && (
                <div className="diag-row">
                  <span className="diag-label">File size</span>
                  <span className="diag-value">{formatBytes(input.bytes)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {processed != null && (
          <div className="diag-section">
            <span className="diag-section-title">Processed</span>
            <div className="diag-rows">
              {processed.width != null && processed.height != null && (
                <div className="diag-row">
                  <span className="diag-label">Canvas</span>
                  <span className="diag-value">
                    {processed.width} × {processed.height}
                  </span>
                </div>
              )}
              {(processed.cropped ?? processed.resized ?? processed.transparent ?? processed.quantized) != null && (
                <div className="diag-row">
                  <span className="diag-label">Flags</span>
                  <span className="diag-chips">
                    {processed.cropped && <span className="chip">Cropped</span>}
                    {processed.resized && <span className="chip">Resized</span>}
                    {processed.transparent && <span className="chip">Transparent</span>}
                    {processed.quantized && <span className="chip">Quantized</span>}
                    {!processed.cropped && !processed.resized && !processed.transparent && !processed.quantized && (
                      <span className="diag-none">none</span>
                    )}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {vectormark != null && (
          <div className="diag-section">
            <span className="diag-section-title">Options</span>
            <div className="diag-rows">
              {vectormark.colors != null && (
                <div className="diag-row">
                  <span className="diag-label">Colors</span>
                  <span className="diag-value">{vectormark.colors}</span>
                </div>
              )}
              {vectormark.epsilon != null && (
                <div className="diag-row">
                  <span className="diag-label">Epsilon</span>
                  <span className="diag-value">{vectormark.epsilon}</span>
                </div>
              )}
              {vectormark.max_error != null && (
                <div className="diag-row">
                  <span className="diag-label">Max error</span>
                  <span className="diag-value">{vectormark.max_error}</span>
                </div>
              )}
              <div className="diag-row">
                <span className="diag-label">Flatten</span>
                <span className="diag-value">{vectormark.flatten ? "yes" : "no"}</span>
              </div>
              <div className="diag-row">
                <span className="diag-label">Symmetry</span>
                <span className="diag-value">{vectormark.no_symmetry ? "off" : "on"}</span>
              </div>
            </div>
          </div>
        )}

        {output != null && (
          <div className="diag-section">
            <span className="diag-section-title">Output</span>
            <div className="diag-rows">
              {output.element_count != null && (
                <div className="diag-row">
                  <span className="diag-label">Elements</span>
                  <span className="diag-value">{output.element_count}</span>
                </div>
              )}
              {output.svg_bytes != null && (
                <div className="diag-row">
                  <span className="diag-label">SVG size</span>
                  <span className="diag-value">{formatBytes(output.svg_bytes)}</span>
                </div>
              )}
              <div className="diag-row">
                <span className="diag-label">Features</span>
                <span className="diag-chips">
                  {output.has_defs && <span className="chip">Defs</span>}
                  {output.has_paths && <span className="chip">Paths</span>}
                  {output.has_primitives && <span className="chip">Primitives</span>}
                  {output.has_symmetry && <span className="chip chip-accent">Symmetry</span>}
                  {!output.has_defs && !output.has_paths && !output.has_primitives && !output.has_symmetry && (
                    <span className="diag-none">none</span>
                  )}
                </span>
              </div>
            </div>
          </div>
        )}

        {warnCount > 0 && (
          <div className="diag-section diag-section-warn">
            <span className="diag-section-title">Warnings</span>
            <ul className="diag-warnings">
              {warnings!.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </details>
  );
}

// ─── Empty / Status panels ────────────────────────────────────────────────────

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
