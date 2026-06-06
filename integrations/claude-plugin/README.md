# vectormark-logo-pipeline (Claude Code plugin)

A Claude Code plugin that wires up a **generate → idealize** logo pipeline:

1. **Generate** a raster logo with whatever image-generation tooling the agent
   has (an image model/API, the Canva MCP, a local diffusion endpoint — it's
   tool-agnostic).
2. **Idealize** that raster into a clean, structured, exactly-symmetric SVG by
   piping it through the [`vectormark`](../../README.md) CLI.

The deliverable is always the **idealized SVG**, not a raw PNG or tracer dump.

## What's in here

```
integrations/claude-plugin/
├── .claude-plugin/
│   └── plugin.json              # plugin manifest
├── commands/
│   └── generate-logo.md         # /generate-logo slash command (thin wrapper)
├── skills/
│   └── generate-logo/
│       ├── SKILL.md             # primary skill: triggers + workflow
│       └── reference.md         # full CLI surface, tool guidance, troubleshooting
└── README.md                    # this file
```

## Requirements

- **Claude Code** (or another agent runtime that loads Claude Code plugins/skills).
- An **image-generation tool** available to the agent (optional but expected —
  without one, the agent will ask you to supply a raster).
- The **`vectormark` CLI** installed, or a checkout of this repo so the agent
  can run `uv run vectormark`.

Install vectormark:

```bash
# from a checkout of this repo
uv sync --extra dev          # then: uv run vectormark ...
# or into the active environment
uv pip install vectormark    # or: pip install vectormark
```

## Install the plugin

This plugin ships inside the vectormark repo at `integrations/claude-plugin/`.
You can use it a few ways:

**A. Local / dev (point Claude Code at the directory)**

Add it to your Claude Code plugins config (or symlink it under your plugins
dir) so the skill and command are discovered:

```bash
# example: symlink into your user plugins cache layout
ln -s "$(pwd)/integrations/claude-plugin" ~/.claude/plugins/cache/vectormark-logo-pipeline
```

**B. Marketplace entry (recommended for distribution)**

Publish a `marketplace.json` that points at this subdirectory, then
`/plugin marketplace add <url>` and install `vectormark-logo-pipeline`. Example
marketplace entry source block:

```json
{
  "name": "vectormark-logo-pipeline",
  "source": {
    "source": "git-subdir",
    "url": "https://github.com/pradeepmouli/vectormark.git",
    "path": "integrations/claude-plugin",
    "ref": "main"
  }
}
```

> See **Open decisions** below — bundled-subdir vs. its own repo vs. a
> marketplace entry is a packaging choice the maintainer should ratify.

## Usage

Once installed, either:

- Just ask: *"Generate a minimal two-color fox logo as an SVG."* The
  `generate-logo` skill triggers automatically.
- Or run the slash command: `/generate-logo a minimal two-color fox, geometric`

The agent will generate a raster, run `vectormark <raster> -o <out>.svg`, and
hand you the saved SVG path. It targets **flat-color segmented marks** (logos,
icons, emblems), not photos or full illustrations.

## How the handoff works

The skill writes a local raster (e.g. `./logo-raster.png`) then runs:

```bash
vectormark ./logo-raster.png -o ./logo.svg        # or: uv run vectormark ...
```

Useful flags: `--colors N`, `--no-symmetry`, `--flatten`, `--epsilon`,
`--max-error`. Full details live in
[`skills/generate-logo/reference.md`](skills/generate-logo/reference.md).

## Open decisions (for the maintainer)

- **Packaging**: bundled here as `integrations/claude-plugin/` (default).
  Alternatives: extract to its own repo, or only expose via a marketplace entry.
- **Default generation tool**: intentionally tool-agnostic. If you want a
  canonical default (e.g. the Canva MCP), say so and the skill can lead with it.
- **Slash command**: included (`/generate-logo`) as a thin wrapper; drop it if
  you prefer skill-only triggering.
- **Distribution**: no `marketplace.json` is committed yet; add one when you
  decide on the channel.

## License

MIT (same as vectormark).
