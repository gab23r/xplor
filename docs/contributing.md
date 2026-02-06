# Contributing

Thank you for your interest in contributing to `xplor`! We welcome contributions from the community, whether it's reporting bugs, suggesting features, improving documentation, or submitting code.

## 🤝 How to Contribute

### 🐛 Reporting Issues

If you find a bug, encounter an unexpected behavior, or have a suggestion for improvement:

1.  **Check Existing Issues:** Before opening a new issue, please search the [issue tracker](https://github.com/gab23r/xplor/issues) to see if the issue or suggestion already exists.
2.  **Create a New Issue:** If not found, create a new issue with a clear, descriptive title.
3.  **Provide Context:**
      * **For Bugs:** Include the version of Python, Polars, and the solver you are using (Gurobi/OR-Tools). Provide clear, minimal steps and code to reproduce the unexpected behavior.
      * **For Features:** Clearly outline the use case and the benefits of the proposed feature.

### 💻 Submitting Pull Requests (PRs)

1.  **Fork** the `xplor` repository.
2.  **Clone** your fork locally and navigate to the directory.
3.  **Create a new branch** for your changes:
    ```bash
    git checkout -b feature/brief-description-of-feature
    # or
    git checkout -b fix/issue-number-or-bug-name
    ```
4.  **Implement your changes.**
5.  **Add tests** for new functionality or to verify bug fixes.
6.  **Commit** your changes using clear, descriptive messages (conventional commits are appreciated but not strictly required).
7.  **Push** to your fork and [submit a pull request] to the main repository's `main` branch.

## 🛠️ Development Setup & Checks

To ensure consistency and quality, we rely on a few specific tools.

### Development Environment

The setup includes installing the optional dependencies needed for testing multiple backends (Gurobi, OR-Tools).

```bash
# Clone your fork (replace YOUR-USERNAME)
git clone https://github.com/YOUR-USERNAME/xplor.git
cd xplor

# Install development dependencies
uv sync

# Install and run the pre-commit hooks
uv run prek install

# Run test
uv run pytest

# Build the documentation
mkdocs build

# Serve docs locally
mkdocs serve
```

### Task Runner (`just`)

This project uses [`just`](https://github.com/casey/just) as a command runner to simplify common development tasks.

**Installation:**

```bash
# macOS
brew install just

# Linux
# See https://github.com/casey/just#installation

# Or via cargo
cargo install just
```

**Available commands:**

```bash
just install       # Install dependencies
just test          # Run tests
just test-cov      # Run tests with coverage report
just lint          # Check code with ruff
just fix           # Auto-fix linting and formatting issues
just prek          # Run pre-commit hooks
just build         # Build the package
just clean         # Remove build artifacts and cache
just bump <type>   # Bump version, commit, push and publish (patch|minor|major)
just help          # List all available commands
```

## Questions?

Feel free to open an issue for questions or discussions about contributing.
