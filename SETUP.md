# Setup Guide

## Installation Options

### Option 1: Using pip (Standard Python Environment)

If you're in a standard Python environment:

```bash
cd we_explore_context
pip install -r requirements.txt
```

### Option 2: Nix Environment (Firebase Studio / NixOS)

If you're in a Nix/Firebase Studio environment, you have two options:

#### A. Add to .idx/dev.nix (Recommended - Persistent)

Edit `.idx/dev.nix` and add:

```nix
packages = [
  pkgs.python311
  pkgs.python311Packages.pip
  pkgs.python311Packages.pydantic
  pkgs.python311Packages.openai
  pkgs.python311Packages.numpy
  pkgs.python311Packages.sentence-transformers
];
```

Then restart the workspace.

#### B. Use nix-env (Quick but ephemeral)

```bash
# Install Python packages via nix-env
nix-env -iA nixpkgs.python311Packages.pydantic
nix-env -iA nixpkgs.python311Packages.openai
nix-env -iA nixpkgs.python311Packages.numpy

# Add to PATH
export PATH="$HOME/.nix-profile/bin:$PATH"
export PYTHONPATH="$HOME/.nix-profile/lib/python3.11/site-packages:$PYTHONPATH"
```

Note: This is temporary and won't persist across workspace restarts.

### Option 3: Python Virtual Environment

```bash
cd we_explore_context

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env` file or create one:

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=YourApp
```

2. Get an OpenRouter API key from: https://openrouter.ai/

## Verifying Installation

Test that dependencies are installed:

```bash
python3 -c "import pydantic, openai, numpy; print('✓ All dependencies available')"
```

Test that the framework can be imported:

```bash
python3 -c "from src.context_engine import Agent; print('✓ Context Engine loads correctly')"
```

## Running Examples

Once dependencies are installed:

```bash
# Run any example
python3 examples/01_basic_agent.py
python3 examples/02_layered_memory.py
python3 examples/03_multi_agent.py
python3 examples/04_self_baking.py
python3 examples/05_research_assistant.py
```

## Troubleshooting

### ModuleNotFoundError: No module named 'pydantic'

Dependencies not installed. Follow installation instructions above.

### API Key Error

Make sure `.env` file exists with valid `OPENROUTER_API_KEY`.

### Import Error from src.context_engine

Make sure you're running from the project root directory.

## Minimal Requirements

- Python 3.11+
- OpenRouter API key (free tier available)
- ~500MB disk space for dependencies
