# UV Package Manager Guide

Figureland fully supports UV as a modern, fast alternative to pip.

## Installation with UV

### 1. Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create project environment
```bash
# Initialize new project
uv init my-ml-project
cd my-ml-project
```

### 3. Add Figureland dependency
```bash
uv add figureland

# With optional extras
uv add "figureland[cloud]"
uv add "figureland[dev,cloud]"
```

### 4. Run commands
```bash
# Run figureland CLI
uv run figureland

# Run with parameters
uv run figureland resolution=[512,512] n_episodes=1000 parallel_generation=true
```

## Pyenv + UV Workflow

```bash
# Set Python version with pyenv
pyenv install 3.10.13
pyenv local 3.10.13

# Create virtual environment
uv venv --python 3.10.13
source .venv/bin/activate

# Install dependencies
uv add figureland
```

## UV Advantages

- **10-100x faster** dependency resolution than pip
- Deterministic lockfiles
- Built-in virtual environment management
- Global package cache
- Proper dependency resolution

## Install Pre-release versions
```bash
uv add --prerelease=allow figureland
```

## Install from GitHub
```bash
uv add git+https://github.com/figureland/figureland.git
```

## Run without installing
```bash
uvx figureland --help
```

For more information: https://github.com/astral-sh/uv
