# Contributing to Vic-Torch

Thank you for your interest in contributing to Vic-Torch! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run tests** to ensure everything works: `pytest`

## Development Workflow

1. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add or update tests as needed
4. Run tests to ensure they pass: `pytest`
5. Commit your changes with clear, descriptive commit messages
6. Push to your fork and submit a pull request

## Code Style

- Follow existing code conventions in the repository
- Keep code modular and well-documented
- Write clear commit messages that explain the "why" behind changes

## Testing

- All new features should include appropriate tests
- Tests should be defensive and skip gracefully if dependencies are missing
- Run the full test suite before submitting a PR: `pytest tests/`

## Pull Request Guidelines

- Provide a clear description of the problem and solution
- Reference any related issues
- Ensure all tests pass
- Keep PRs focused on a single concern when possible

## Reproducibility

When working with training scripts or experiments:
- Use deterministic seeds (configured in `configs/default.yaml`)
- Document any changes to model architecture or training procedures
- Save checkpoints using the provided checkpoint utilities

## Questions?

If you have questions about contributing, please open an issue for discussion.

## License

By contributing to Vic-Torch, you acknowledge that your contributions will be subject to the project's LICENSE terms.
