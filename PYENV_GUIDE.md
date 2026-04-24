# Pyenv Installation Guide

Figureland works perfectly with pyenv for Python version management.

## Installation with Pyenv

### 1. Install pyenv
```bash
curl https://pyenv.run | bash
```

Add to your shell config (~/.bashrc, ~/.zshrc):
```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### 2. Install supported Python version
```bash
pyenv install 3.10.13
pyenv global 3.10.13
```

### 3. Create virtual environment
```bash
pyenv virtualenv 3.10.13 figureland
pyenv activate figureland
```

### 4. Install Figureland
```bash
pip install figureland
```

Or for development:
```bash
git clone https://github.com/figureland/figureland.git
cd figureland
pip install -e ".[dev,cloud]"
```

## Pyenv Best Practices

- Always use a dedicated virtual environment per project
- Use `pyenv local figureland` in project directory
- Pin Python version in `.python-version` file:
  ```bash
  echo "figureland" > .python-version
  ```

## Test Installation
```bash
figureland --help
figureland --version
```

## Troubleshooting

If you encounter issues:
```bash
# Update pyenv
pyenv update

# Reinstall Python
pyenv uninstall 3.10.13
pyenv install 3.10.13

# Verify installation
pyenv versions
which python
python --version
```
