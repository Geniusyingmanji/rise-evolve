---
name: codex-gpt55-dispatch
description: Use when needing free GPT-5.5 on this machine — for batch data annotation, codegen via codex agent, or VLM image QC — or when codex calls return alien session memory, empty prompts (input_tokens=2), 403 Public access disabled, or exit code 143 with no artifacts.
---

# Dispatching free GPT-5.5 (codex / direct API)

## Overview
This machine has key-free GPT-5.5 via a local Copilot proxy at `localhost:4142`. Three access modes: codex agent (codegen), direct responses API (batch annotation), vision API (image QC). Several lookalike routes are broken traps.

## The one working route

| Mode | Command |
|---|---|
| codex agent (codegen) | `/home/azureuser/.nvm/versions/node/v18.20.8/bin/codex exec --skip-git-repo-check -c model_provider=github -m gpt-5.5 -c model_reasoning_effort=high -c sandbox_mode=workspace-write "PROMPT"` |
| batch text API | `POST http://localhost:4142/v1/responses` with `Authorization: Bearer dummy`, body `{"model":"gpt-5.5","reasoning":{"effort":"low"},"input":[{"role":"user","content":[{"type":"input_text","text":"..."}]}]}` |
| vision API | same, add `{"type":"input_image","image_url":"data:image/png;base64,..."}` to content |

Parse replies from `output[].content[]` where `type=="output_text"`. Working reference implementations: `scripts/data/refine_tasks_gpt55.py` (batch + retries + resume), `scripts/data/vlm_verify_renders.py` (vision).

## Traps (each one cost a real failure)

| Trap | Symptom | Fix |
|---|---|---|
| `codex` on PATH is a hermes wrapper (`cc-connect-server/bin/hermes-codex`) | prompt arrives empty (`input_tokens: 2`), reply contains other projects' memory, thread_id `claude:...` | use the real binary path above |
| `codex-azure` / `azure_uami` profile (proxy :9876) | `403 Public access is disabled` — Azure resource blocks public net; unfixable locally | use the 4142 route |
| `/v1/chat/completions` on 4142 | `model "gpt-5.5" is not accessible via /chat/completions` | use `/v1/responses` |
| `timeout N codex exec ...` too small for big codegen | exit code 143, **zero artifacts** | budget ≥5400s for multi-file codegen; prefer `model_reasoning_effort=medium` for speed; split huge jobs |
| default reasoning effort is xhigh (config.toml) | slow batch calls | always set effort explicitly: `low` for annotation, `medium/high` for codegen |

## Codegen prompt rules (what made codex output land correctly)
- Demand a SELF-TEST in the prompt ("run with --limit N, assert X, print summary") — codex then debugs itself.
- Name files/dirs it must NOT touch (running pipelines read them); route new facts to sidecar files.
- For batch concurrency: 8-10 workers sustained ~0.7-0.9 tasks/s; the proxy tolerates ~16-20 concurrent.
- **Decompose before delegating**: a task requiring codex to reverse-engineer existing complex code (e.g. "recompute facts by importing the generators' solvers") burned 2×90min with zero artifacts. Survey the data YOURSELF first — often the needed facts are already stored in the rows — then hand codex a mechanical spec with exact field names/mappings. Mechanical specs finish in minutes; archaeology specs time out.

## Red flags
- Reply mentions unfamiliar projects → you hit the hermes wrapper.
- Long codex run with no file changes → check it wasn't timeout-killed (143).

## Notes
- **ALWAYS redirect stdin: `codex exec ... < /dev/null`** when launching via nohup/background. `Reading additional input from stdin...` means codex waits for stdin EOF — in a backgrounded pipe context stdin never closes and codex hangs FOREVER producing zero output (cost three 45-90min timeout-kills before diagnosis). Interactively it's harmless because stdin closes fast.
- Timeout budgets: simple single-reply codex calls finish in <5 min (300s is plenty); multi-file codegen ≥3600s.
