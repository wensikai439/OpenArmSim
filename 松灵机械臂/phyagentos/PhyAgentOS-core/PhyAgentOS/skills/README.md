# PhyAgentOS Skills

This directory contains built-in skills that extend PhyAgentOS's capabilities.
Forge Skills are installed and managed separately. Concrete Forge Skill bundles, executable nodes,
models, and simulation assets are not included in this source distribution.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:

- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

Workspace Skills under `<workspace>/skills/` take precedence over built-ins with the same name.

## Activation and learned workflows

With `agents.evolution.enabled=true`, the Agent activates a matching registered Skill through `activate_skill` before execution. Activation returns the full workflow and only the active scoped Lessons that apply to the current task. A direct file read does not establish task-to-Skill attribution.

A turn may use one primary Skill and multiple supporting Skills. Only the primary can receive an automatic managed-workflow update. Built-in Skills are never edited in place; a promoted revision is written as a workspace override. Learned content is enclosed by PAOS managed-block markers and always remains non-always-on.

Skill-bound Lesson projections live at `<workspace>/skills/<name>/references/LESSONS.md`. They are generated for review from the experience ledger and should not be treated as global constraints or edited as the source of truth.

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `github` | Interact with GitHub using the `gh` CLI |
| `weather` | Get weather info using wttr.in and Open-Meteo |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `tmux` | Remote-control tmux sessions |
| `clawhub` | Search and install skills from ClawHub registry |
| `skill-creator` | Create new skills |
