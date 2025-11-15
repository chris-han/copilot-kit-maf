# CopilotKit <> Microsoft Agent Framework (Python)

This starter pairs the [CopilotKit](https://copilotkit.ai) frontend runtime with the [Microsoft Agent Framework (MAF)](https://aka.ms/agent-framework) over the AG-UI protocol. You get:

- A Next.js 15 UI that demonstrates shared state, frontend actions, generative UI, and human-in-the-loop flows.
- A FastAPI server powered by Microsoft Agent Framework and `agent-framework-ag-ui`.
- Scripts that spin up everything locally so you can copy/paste the pattern into your own applications.

## Prerequisites

- Node.js 20+ and your preferred package manager (examples below use pnpm)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- Either:
  - **OpenAI** – `OPENAI_API_KEY` and `OPENAI_CHAT_MODEL_ID`
  - **Azure OpenAI** – `MAF_PROVIDER=azure`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, plus either `AZURE_OPENAI_API_KEY` or `az login`

> **Lockfiles:** JS lockfiles are ignored so teams can pick pnpm/yarn/npm/bun without conflicts. Commit your own if your workflow needs it.

## Getting Started

1. **Install frontend dependencies**

   ```bash
   pnpm install    # or npm install / yarn install / bun install
   ```

   The `postinstall` hook runs `uv sync` inside `agent/` so Microsoft Agent Framework packages are ready. Need to rerun manually? `npm run install:agent`.

2. **Configure the agent**

   Create `agent/.env` with the credentials for your provider.

   **OpenAI example**

   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_CHAT_MODEL_ID=gpt-4o-mini
   ```

   **Azure OpenAI example**

   ```env
   MAF_PROVIDER=azure
   AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
   # If you are not relying on az login:
   # AZURE_OPENAI_API_KEY=...
   ```

3. **Run everything**

   ```bash
   pnpm dev
   ```

   This launches both the Next.js UI (`http://localhost:3000`) and the FastAPI + MAF backend (`http://localhost:8000`).

## Available Scripts

| Script         | Description                                                                          |
| -------------- | ------------------------------------------------------------------------------------ |
| `dev`          | Runs the Next.js UI and the Microsoft Agent Framework server                         |
| `dev:debug`    | Same as `dev`, but with verbose logging                                              |
| `dev:ui`       | Starts only the Next.js app                                                          |
| `dev:agent`    | Starts only the MAF FastAPI server (`uv run src/main.py`)                            |
| `build`        | Builds the Next.js app                                                               |
| `start`        | Runs the built Next.js app in production mode                                        |
| `lint`         | Runs ESLint                                                                          |
| `install:agent`| Re-syncs the Python virtual environment using uv                                     |

## Agent Structure

- `agent/src/agent.py` – defines the Microsoft Agent Framework `ChatAgent`, tools, and shared-state schema.  
- `agent/src/main.py` – exposes that agent through FastAPI with `add_agent_framework_fastapi_endpoint`.  
- `agent/pyproject.toml` & `uv.lock` – lock Microsoft Agent Framework + AG-UI dependencies.

Highlights:

- **Shared state:** the proverb list stays synced with the UI via AG-UI state events.  
- **Frontend actions:** `get_weather` renders a weather card entirely from agent-side tool calls.  
- **Human in the loop:** `go_to_moon` requires approval, showcasing `approval_mode="always_require"`.

## Updating the UI

The main CopilotKit experience lives in `src/app/page.tsx`. Feel free to:

- Change or add cards in `src/components/`.
- Register new frontend actions with `useCopilotAction`.
- Customize the Copilot Sidebar theme and suggestions.

## Troubleshooting

- **“I’m having trouble connecting to my tools”**
  1. Make sure the FastAPI server (`uv run src/main.py`) is running on `http://localhost:8000`.
  2. Confirm the URL in `src/app/api/copilotkit/route.ts` points to the correct agent endpoint.
  3. Verify your environment variables—the Microsoft Agent Framework client will fail fast if credentials are missing.

- **Python dependency hiccups**

  ```bash
  cd agent
  uv sync
  uv run src/main.py
  ```

- **Switching providers**
  - Set/unset `MAF_PROVIDER=azure` inside `agent/.env`.
  - Restart `pnpm dev`.
  - If you rely on `az login`, run it before starting the server.

## Useful Links

- [Microsoft Agent Framework](https://aka.ms/agent-framework)
- [CopilotKit Documentation](https://docs.copilotkit.ai)
- [Build a frontend for your Microsoft Agent Framework agents with AG-UI](https://www.copilotkit.ai/blog/build-a-frontend-for-your-microsoft-agent-framework-agents-with-ag-ui)

## License

MIT – see `LICENSE` for details.
