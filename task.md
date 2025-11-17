# CI Setup Checklist for Solo Developers

Quick checklist to fix CI issues and get your repo pushing smoothly.

## ✅ Completion Status

**Completed Items:**
- ✅ Created `.python-version` file (Python 3.12)
- ✅ Created `.github/workflows/ci.yml` workflow
- ✅ Created `mypy.ini` configuration file
- ✅ Added ruff configuration to `pyproject.toml`
- ✅ Verified `.python-version` is not in `.gitignore`

**Notes:**
- CI workflow uses Python 3.12 from `.python-version` file
- Tests are blocking; lint and type checks are non-blocking
- Project supports Python 3.9+ but CI runs on 3.12
- Pre-commit hooks remain configured with black/flake8 (ruff config added for optional use)

## 1. Python Version File

```bash
# Create .python-version file (if using Python 3.12+)
echo "3.12" > .python-version

# OR if using a different version
echo "3.11" > .python-version  # adjust as needed

# Make sure it's NOT in .gitignore
# If it is, remove this line from .gitignore:
# .python-version
```

**Check:**
- [x] `.python-version` file exists
- [x] `.python-version` is tracked in git (not in .gitignore)

## 2. Update CI Workflow

✅ **COMPLETED**: Created `.github/workflows/ci.yml`

Edit `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'  # OR use: python-version-file: ".python-version"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Test
        run: |
          pytest -q --maxfail=1  # Only require tests to pass

      # Optional: non-blocking checks
      - name: Lint check (non-blocking)
        continue-on-error: true
        run: |
          ruff check . || echo "Linting issues found (non-blocking)"

      - name: Type check (non-blocking)
        continue-on-error: true
        run: |
          mypy . || echo "Type check issues found (non-blocking)"
```

**Key changes:**
- Remove Node.js checks if Python-only project
- Only require tests to pass (blocking)
- Make lint/type checks non-blocking with `continue-on-error: true`

## 3. Configure Mypy (if using)

✅ **COMPLETED**: Created `mypy.ini`

Create/update `mypy.ini`:

```ini
[mypy]
python_version = 3.12
ignore_missing_imports = True
exclude = (site-packages|\.venv|venv|examples|benchmarks|__pycache__)
follow_imports = skip  # Don't check third-party packages

# Ignore known problematic modules
[mypy-pytest.*]
follow_imports = skip
ignore_errors = True

[mypy-_pytest.*]
follow_imports = skip
ignore_errors = True

# Add other problematic modules as needed
[mypy-your_module_name]
ignore_errors = True
```

## 4. Configure Ruff (if using)

✅ **COMPLETED**: Added ruff configuration to `pyproject.toml`

In `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"  # Match your Python version
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".eggs",
    "*.egg",
    "build",
    "dist",
    "examples",
    "benchmarks",
]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = [
    "E402",  # Module level import not at top (if intentional)
    "E501",  # Line too long (handled by formatter)
    "N806",  # Variable should be lowercase (if using stats conventions)
    "N803",  # Argument should be lowercase (if using stats conventions)
    # Add ignores as needed for your codebase
]
```

## 5. Update Python Requirements

✅ **NOTE**: Project supports Python 3.9+ (configured in pyproject.toml). CI uses Python 3.12 from .python-version file.

In `pyproject.toml`:

```toml
[project]
requires-python = ">=3.12"  # Update to match your target version

classifiers = [
    # ... other classifiers ...
    "Programming Language :: Python :: 3.12",  # Only list versions you support
]
```

## 6. Simplify Pre-commit (Optional)

Edit `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff-format  # Only auto-format, don't block on linting
      - id: ruff
        args: ['--exit-non-zero-on-fix']
        stages: []  # Disabled - won't run automatically

# Remove mypy/pytest hooks if you don't want them blocking commits
```

**Install pre-commit hooks:**
```bash
pre-commit install  # Optional - only if you want auto-formatting
```

## 7. Fix Badges in README

Update badges to match your workflow names:

```markdown
[![Python versions](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/yourusername/yourrepo/workflows/CI/badge.svg)](https://github.com/yourusername/yourrepo/actions)
```

**Key:** Match the workflow name exactly (case-sensitive).

## 8. Quick Test Locally

```bash
# Test that CI checks work locally
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q  # Should pass

# Optional: test lint/type checks
ruff check .
mypy .  # or mypy your_package tests
```

## 9. Commit and Push

```bash
git add .
git commit -m "Fix CI: configure Python version, simplify checks"
git push
```

## Common Issues & Fixes

**Issue: "python-version-file doesn't exist"**
- Create `.python-version` file with your Python version
- Remove `.python-version` from `.gitignore` if it's there

**Issue: "Node.js checks failing (Python-only project)"**
- Remove Node.js matrix entries from CI workflow
- Delete `.nvmrc` if not needed

**Issue: "Mypy checking pytest files with syntax errors"**
- Add `follow_imports = skip` for pytest modules in `mypy.ini`
- Or use `--exclude` flag to exclude site-packages

**Issue: "Ruff complaining about modern syntax"**
- If requiring Python 3.10+, remove `UP045`, `UP006`, `UP007` from ignore list
- Run `ruff check --fix .` to auto-fix
- If requiring Python 3.8+, keep those in ignore list

**Issue: "Pre-commit hooks blocking commits"**
- Disable problematic hooks in `.pre-commit-config.yaml` (set `stages: []`)
- Or use `git commit --no-verify` if hooks are too strict

## Philosophy for Solo Developers

- **Tests must pass** (ensures code works)
- **Everything else is optional** (non-blocking checks)
- **Auto-formatting is nice** (keeps code clean without blocking)
- **Focus on code that works**, not perfect linting
