# Publishing to PyPI

Step-by-step guide to publish Figureland to the official Python Package Index.

## Prerequisites

1. Create PyPI account: https://pypi.org/account/register/
2. Create TestPyPI account: https://test.pypi.org/account/register/
3. Install required tools:
   ```bash
   # Using uv
   uv add --dev build twine

   # Using pip
   pip install build twine
   ```

## Step 1: Verify Package Configuration

Check all metadata is correct:
```bash
# Check pyproject.toml
uv run python -m build --dry-run
```

## Step 2: Build Distribution Packages

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source and wheel distributions
uv run python -m build
```

This creates:
- `dist/figureland-0.1.0.tar.gz` (source distribution)
- `dist/figureland-0.1.0-py3-none-any.whl` (pure Python wheel)

## Step 3: Verify Build

Check package contents:
```bash
# List files in wheel
unzip -l dist/*.whl

# Check package metadata
uv run twine check dist/*
```

✅ All checks should pass with no warnings or errors.

## Step 4: Publish to TestPyPI

Always test on TestPyPI first:
```bash
uv run twine upload --repository testpypi dist/*
```

Test installation:
```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ figureland

# Verify installation works
figureland --help
python -c "import figureland; print(figureland.__version__)"
```

## Step 5: Publish to PyPI

Once testing is complete:
```bash
uv run twine upload dist/*
```

## Step 6: Verify Public Installation

```bash
# Install from PyPI
pip install figureland

# Test CLI
figureland --version

# Test import
python -c "
from figureland import DatasetGenerator
import hydra
print('✓ Figureland installed successfully')
"
```

## UV Publishing Alternative

For even faster publishing with uv:
```bash
# Build and publish in one command
uv publish
```

## Post-Publish Checklist

✅ Package appears on PyPI: https://pypi.org/project/figureland/
✅ Installation works on clean environments
✅ CLI command `figureland` is available
✅ All dependencies install correctly
✅ Hydra config files are properly bundled
✅ Documentation links work

## Version Bumping

Before next release:
1. Update version in `pyproject.toml`
2. Update version in `figureland/__init__.py`
3. Add release notes to CHANGELOG.md
4. Create git tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`

## Troubleshooting

**Invalid package name**: Ensure name is all lowercase with no spaces
**File not included**: Check `MANIFEST.in` and `package_data`
**403 Forbidden**: Verify PyPI credentials and permissions
**Installation fails**: Check Python version compatibility and dependencies

## Security Notes

- Never commit your PyPI password to version control
- Use API tokens instead of passwords: https://pypi.org/help/#apitoken
- Enable 2FA for your PyPI account
- Verify package signatures after upload
