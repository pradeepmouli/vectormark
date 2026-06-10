# vectormark MCP integration

vectormark can be exposed as either a local MCP server or an MCP App. The
server exposes the deterministic Python pipeline as tools; the app adds a React
iframe resource for UI-capable MCP hosts.

## Local MCP server

Install the project and run the stdio server:

```bash
uv sync --extra dev
npm --prefix integrations/mcp-app install
npm --prefix integrations/mcp-app run build
uv run vectormark-mcp
```

The server exposes two tools:

- `idealize_logo` converts a local raster logo into structured SVG.
- `render_idealized_logo` renders an existing `idealize_logo` result in the app
  resource. Text-only clients can ignore it.

Tool arguments:

- `image_path` - local PNG/JPG path.
- `output_path` - optional SVG output path. Parent directories are created.
- `epsilon` - primitive/polygon recognition tolerance in pixels.
- `max_error` - Bezier fit tolerance in pixels.
- `colors` - maximum palette colors.
- `flatten` - emit plain paths instead of native SVG primitives and `<use>`.
- `no_symmetry` - disable symmetry detection.

`idealize_logo` is also an MCP App tool. UI-capable clients discover the React
view from `_meta.ui.resourceUri`, then read `ui://vectormark/logo-widget.html`.
The resource is served from the built single-file Vite artifact at
`integrations/mcp-app/dist/mcp-app.html`.

Example MCP client configuration:

```json
{
  "mcpServers": {
    "vectormark": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/pmouli/GitHub.nosync/active/py/vectormark",
        "run",
        "vectormark-mcp"
      ]
    }
  }
}
```

## React app source

The app frontend lives in `integrations/mcp-app`:

- `src/mcp-app.tsx` - React MCP Apps view using
  `@modelcontextprotocol/ext-apps/react`.
- `src/styles.css` - host-variable-aware styling for the SVG preview and
  controls.
- `mcp-app.html` - Vite HTML entry point.
- `dist/mcp-app.html` - generated single-file resource served by Python.

Build it with:

```bash
npm --prefix integrations/mcp-app run build
```

## Security: the file tools are local-trust only

`idealize_logo` reads an arbitrary `image_path` and writes an arbitrary
`output_path` on the host, and `render_idealized_logo` renders caller-supplied
SVG in the app iframe (via `innerHTML`). That is fine for the **local stdio
server**, where the tools act with your own permissions on your own machine.

It is **not safe to expose as-is over the network.** Before hosting the server
(see below), you MUST:

- **Confine paths** — resolve `image_path`/`output_path` against an allowed base
  directory and reject anything that escapes it (no arbitrary read/write).
- **Require auth** — put the endpoint behind Cloudflare Access or OAuth (never an
  open `/mcp`).
- **Treat caller SVG as untrusted** — vectormark's own output is script-free, but
  `render_idealized_logo` renders whatever SVG it is handed; sanitize it (or only
  render server-produced SVG) so a hostile `<script>`/`onload` cannot run in the
  widget. The widget CSP (empty `connect`/`resource` domains) limits but does not
  eliminate this.

## Hosted MCP app

A hosted app has three parts:

1. An MCP server reachable at an HTTPS endpoint such as
   `https://vectormark.daikonic.dev/mcp`.
2. The React UI resource served by the MCP server for previewing the input
   raster and returned SVG.
3. The vectormark compute runtime.

The current implementation depends on Python native/scientific packages
including numpy, scipy, scikit-image, shapely, and Pillow. That makes a pure
Cloudflare Workers implementation the wrong first target: Workers are a
JavaScript/WebAssembly edge runtime, while the existing vectormark pipeline is
CPython plus native wheels.

Recommended Cloudflare shape:

1. Run the Python MCP server as a containerized origin service.
2. Put it behind Cloudflare using `vectormark.daikonic.dev`.
3. Use Cloudflare Access or OAuth before exposing it beyond personal testing.
4. Add an MCP Apps UI resource that calls `idealize_logo` and renders the SVG
   preview.

For a static or routing-only edge layer, a Cloudflare Worker can still be useful
as a thin front door, but it should proxy to a Python origin rather than trying
to run the vectormark pipeline directly.
