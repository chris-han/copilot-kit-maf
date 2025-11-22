# Gemini Context for CopilotKit <> Microsoft Agent Framework (Python)

This document provides an overview of the project, instructions for building and running, and development conventions, derived from the project's `README.md` and initial code exploration. This context is intended to assist in future interactions with the Gemini CLI.

## Project Overview

This project serves as a starter template for integrating CopilotKit experiences with the Microsoft Agent Framework. It features a Next.js frontend for the user interface and a FastAPI backend that exposes a Microsoft Agent Framework agent via the AG-UI protocol. The agent leverages either OpenAI or Azure OpenAI for its AI capabilities.

**Key Technologies:**
*   **Frontend:** Next.js (React)
*   **Backend:** FastAPI (Python)
*   **AI Framework:** Microsoft Agent Framework
*   **AI Providers:** OpenAI or Azure OpenAI
*   **Package Managers:** pnpm (recommended), npm, yarn, bun

**Architecture:**
The application consists of a Next.js UI that communicates with a FastAPI server. The FastAPI server hosts the Microsoft Agent Framework agent, which interacts with AI models (OpenAI/Azure OpenAI) and exposes its functionalities over the AG-UI protocol.

## Building and Running

### Prerequisites

*   OpenAI or Azure OpenAI credentials
*   Python 3.12+
*   `uv` (Python package manager)
*   Node.js 20+
*   One of the following JavaScript package managers: pnpm (recommended), npm, yarn, or bun.

### Installation

1.  **Install JavaScript/TypeScript dependencies:**
    Use your preferred package manager. The `README.md` recommends `pnpm`.
    ```bash
    # Using pnpm (recommended)
    pnpm install

    # Using npm
    npm install

    # Using yarn
    yarn install

    # Using bun
    bun install
    ```
    > **Note:** This command automatically sets up the Python environment. If manual intervention is needed for the Python agent, run:
    > ```bash
    > npm run install:agent
    > ```

2.  **Set up Agent Credentials:**
    Create a `.env` file inside the `agent` folder.
    *   **For OpenAI:**
        ```
        OPENAI_API_KEY=sk-...your-openai-key-here...
        OPENAI_CHAT_MODEL_ID=gpt-4o-mini
        ```
    *   **For Azure OpenAI:**
        ```
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
        # Optional: If not relying on az login
        # AZURE_OPENAI_API_KEY=...
        ```

### Running the Development Server

To start both the UI and the Microsoft Agent Framework server concurrently:

```bash
# Using pnpm
pnpm dev

# Using npm
npm run dev

# Using yarn
yarn dev

# Using bun
bun run dev
```

### Available Scripts

The following scripts are defined in `package.json` and can be run using your preferred package manager (e.g., `npm run <script-name>`):

*   `dev`: Starts both UI and agent servers in development mode.
*   `dev:debug`: Starts development servers with debug logging enabled.
*   `dev:ui`: Starts only the Next.js UI server.
*   `dev:agent`: Starts only the Microsoft Agent Framework server.
*   `build`: Builds the Next.js application for production.
*   `start`: Starts the production server.
*   `lint`: Runs ESLint for code linting.
*   `install:agent`: Installs Python dependencies for the agent.

## Development Conventions

*   **Linting:** ESLint is used for code linting (`npm run lint`).
*   **Extensibility:** The starter is designed to be easily extensible and customizable.
*   **Main UI Component:** The primary UI component is located at `src/app/page.tsx`.

## Troubleshooting

### Agent Connection Issues

If you encounter "I'm having trouble connecting to my tools", ensure:

1.  The Microsoft Agent Framework agent is running on port 8880.
2.  Your OpenAI/Azure credentials are set correctly in the `agent/.env` file.
3.  Both the UI and agent servers started successfully.

### Python Dependencies

If you experience Python import errors, navigate to the `agent` directory and run:

```bash
cd agent
uv sync
uv run src/main.py
```
