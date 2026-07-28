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

The server exposes three tools:

- **`idealize_logo({image, options})`** — primary file-first tool. Pass the image
  by reference: a ChatGPT/host file (`download_url`+`file_id`), a local `path`, an
  HTTPS `url`, a `data_uri`, or raw `base64`. The server resolves and preprocesses
  the reference; no client-side base64 encoding is required. The tool is marked
  `openai/fileParams` so ChatGPT-generated or uploaded images can be passed
  directly as file references. Returns structured SVG, dimensions, diagnostics, and
  a best-effort PNG preview block.

  > For ChatGPT-generated or uploaded images, pass the image/file reference
  > directly (the tool is marked `openai/fileParams`). Do not base64-encode
  > unless using a fallback host that cannot provide file references.

- **`idealize_logo_data`** *(DEPRECATED fallback)* — accepts a bare base64 string
  or a `data:image/...;base64,...` URI. Use only on hosts that cannot provide file
  references; prefer `idealize_logo` everywhere else.

- **`render_idealized_logo({result})`** — renders an existing `idealize_logo` result
  in the app widget. Pass the whole result object returned by `idealize_logo`;
  `svg_bytes` is re-derived server-side from the SVG (never trusted from the caller).
  Text-only clients can ignore this tool.

Tool options (passed inside the `options` dict for `idealize_logo`):

- `epsilon` - primitive/polygon recognition tolerance in pixels.
- `max_error` - Bezier fit tolerance in pixels.
- `colors` - maximum palette colors.
- `flatten` - emit plain paths instead of native SVG primitives and `<use>`.
- `no_symmetry` - disable symmetry detection.
- `preprocess.crop_to_content` / `preprocess.max_size_px` / `preprocess.preserve_transparency`
  / `preprocess.quantize` — preprocessing controls.

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
- **Treat caller SVG as untrusted** — `render_idealized_logo` echoes whatever SVG
  it is handed. The bundled app renders the preview via an `<img>` data URI (an
  `<img>`-loaded SVG cannot execute scripts), so hostile `<script>`/`onload` cannot
  run in the widget; still strip scripts/event-handlers/external refs server-side as
  defense-in-depth for any non-`<img>` consumer. The widget CSP (empty
  `connect`/`resource` domains) limits but does not eliminate this.

## ChatGPT desktop (remote HTTP transport)

ChatGPT does not launch local stdio servers — it connects to a **remote HTTPS MCP
server** and renders the widget from the `openai/outputTemplate` meta the tools
already carry. So expose vectormark over HTTP and give ChatGPT an HTTPS URL.

The server runs an HTTP transport when `VECTORMARK_MCP_TRANSPORT` is set
(`streamable-http` or `sse`); `VECTORMARK_MCP_HOST` / `VECTORMARK_MCP_PORT` set the
bind address (default `127.0.0.1:8000`, endpoint at `/mcp`):

```bash
VECTORMARK_MCP_TRANSPORT=streamable-http VECTORMARK_MCP_PORT=8000 \
  uv run --extra server --extra scoring vectormark-mcp
```

Give it a public HTTPS URL (a tunnel is fine for personal testing):

```bash
brew install cloudflared        # one-time
cloudflared tunnel --url http://localhost:8000   # prints https://<name>.trycloudflare.com
```

Then in ChatGPT: **Settings → Connectors → Developer mode → Add** the URL
`https://<name>.trycloudflare.com/mcp`.

**Safe-by-default over HTTP.** Because an HTTP transport can be network-reachable,
host-filesystem access is restricted automatically: `idealize_logo` still works
but rejects `path`-based image references (no arbitrary file read), and
`idealize_logo_data` **ignores `output_path`** (no host writes). Over HTTP, the
primary flow is the ChatGPT file-first flow: ChatGPT passes the image reference
(e.g. `download_url`+`file_id`) directly to `idealize_logo` (no base64 needed);
the widget previews the returned SVG. `idealize_logo_data` remains available as
a fallback for hosts that cannot provide file references. A trycloudflare URL is
unauthenticated, so keep it ephemeral or put Cloudflare Access / OAuth in front for
anything beyond personal testing.

### OpenAI Secure MCP Tunnel (preferred over a public tunnel)

OpenAI's [Secure MCP Tunnels](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
fit better than a public cloudflared/ngrok URL. The connection is **outbound-only**:
a local `tunnel-client` long-polls OpenAI's control plane and forwards each JSON-RPC
request to your private MCP server, so nothing is exposed to the open internet and no
inbound firewall rule is needed. The "open `/mcp`" risk the security section warns
about goes away — the server stays bound to `localhost` and only OpenAI's
authenticated tunnel reaches it.

Point the tunnel at the **HTTP** server (`--mcp-server-url`), not a stdio server
(`--mcp-command`): over HTTP the filesystem tools are withheld (see "Safe-by-default"
above), so even through the authenticated tunnel ChatGPT only gets `idealize_logo_data`
plus the widget — never arbitrary file read/write.

```bash
# 1. run the private HTTP server (localhost only)
VECTORMARK_MCP_TRANSPORT=streamable-http VECTORMARK_MCP_PORT=8000 \
  uv run --extra server --extra scoring vectormark-mcp        # http://localhost:8000/mcp

# 2. download tunnel-client from github.com/openai/tunnel-client/releases, then init a profile
tunnel-client init --profile vectormark \
  --tunnel-id tunnel_<your-tunnel-id> \
  --mcp-server-url http://localhost:8000/mcp        # writes ~/.config/tunnel-client/vectormark.yaml

# 3. run the daemon (keep it up while you use the connector)
export CONTROL_PLANE_API_KEY="sk-..."               # a Tunnels-scoped runtime key
tunnel-client run --profile vectormark
```

Create the tunnel ID and key under
`https://platform.openai.com/settings/organization/tunnels` and `.../api-keys`. Then,
**while `tunnel-client run` is up**, add/verify the connector in ChatGPT at
`https://chatgpt.com/#settings/Connectors` (Developer mode). The daemon must stay
healthy for connector discovery and for every MCP call ChatGPT makes.

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
