# vectormark-logo-pipeline (Claude Code plugin)

A Claude Code plugin that wires up a **generate → idealize** logo pipeline:

1. **Generate** a raster logo with a raster image-generation tool or model the
   agent has (an image-generation model/API, a diffusion endpoint — it's
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
- A **raster image-generation tool or model** available to the agent (optional
  but expected — without one, the agent will ask you to supply a raster).
- The **`vectormark` CLI** installed, or a checkout of this repo so the agent
  can run `uv run vectormark`.

Install vectormark:

```bash
# from a checkout of this repo
uv sync --extra dev          # then: uv run vectormark ...
# or install the checkout into the active environment (editable)
uv pip install -e .          # or: pip install -e .
```

## Install the plugin

This plugin ships inside the vectormark repo at `integrations/claude-plugin/`,
and the repo doubles as a **plugin marketplace** (a `.claude-plugin/marketplace.json`
at the repo root lists it).

**A. Marketplace (recommended)**

Add the marketplace, then install the plugin:

```text
/plugin marketplace add pradeepmouli/vectormark
/plugin install vectormark-logo-pipeline@vectormark
```

For a reproducible install, pin the marketplace to a published release tag
rather than tracking the default branch, e.g.:

```text
/plugin marketplace add pradeepmouli/vectormark@vectormark-logo-pipeline-v0.1.0
```

(Release tags are cut per plugin version; tracking the default branch always
pulls the latest, which may change under you.)

**B. Local / dev (run against the working tree)**

Point Claude Code at the plugin directory so the skill and command are
discovered from your checkout without installing:

```bash
claude --plugin-dir ./integrations/claude-plugin
```

(Symlinking into `~/.claude/plugins/cache/` does **not** register a plugin —
that cache holds copies written by the installer, so use `--plugin-dir` or the
marketplace flow above.)

## Usage

Once installed, either:

- Just ask: *"Generate a minimal two-color fox logo as an SVG."* The
  `generate-logo` skill triggers automatically.
- Or run the slash command. When installed as a plugin it is namespaced by the
  plugin: `/vectormark-logo-pipeline:generate-logo a minimal two-color fox, geometric`

The agent will generate a raster, run `vectormark <raster> -o <out>.svg`, and
hand you the saved SVG path. It targets **flat-color segmented marks** (logos,
icons, emblems), not photos or full illustrations.

## How the handoff works

The skill writes a raster to a **unique** local path (a slug from the brief,
e.g. `./fox-logo.png` — not a fixed name, so re-runs don't overwrite), then runs:

```bash
vectormark ./fox-logo.png -o ./fox-logo.svg        # or: uv run vectormark ...
```

Useful flags: `--colors N`, `--no-symmetry`, `--flatten`, `--epsilon`,
`--max-error`. Full details live in
[`skills/generate-logo/reference.md`](skills/generate-logo/reference.md).

## License

MIT (same as vectormark).
