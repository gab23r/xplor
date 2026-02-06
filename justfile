# Install dependencies
install:
    uv sync

# Run tests
test:
    uv run pytest

# Run tests with coverage
test-cov:
    uv run pytest --cov=src --cov-report=html --cov-report=term

# Run linter and formatter
lint:
    uv run ruff check .
    uv run ruff format --check .

# Fix linting and formatting issues
fix:
    uv run ruff check --fix .
    uv run ruff format .

ty:
    uv run ty check .

# Run pre-commit hooks
prek:
    uv run prek install
    uv run prek run --all-files

# Build the package
build:
    uv build

# Bump version, commit, tag and push (GitHub Actions will publish to PyPI) (usage: just bump patch|minor|major)
bump type:
    just clean
    uv version --bump {{type}}
    git add pyproject.toml uv.lock
    git commit -m "bump to $(uv version --short)"
    git tag "v$(uv version --short)"
    git push && git push --tags

# Build documentation
docs-build:
    uv run mkdocs build

# Serve documentation locally with live reload
docs-serve:
    uv run mkdocs serve

# Deploy documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy --force

# Clean build artifacts and cache
clean:
    rm -rf dist/ htmlcov/ .pytest_cache/ .ruff_cache/ site/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# List all available commands
help:
    @just --list
