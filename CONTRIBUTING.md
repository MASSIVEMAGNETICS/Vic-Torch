# Contributing to Vic-Torch

Thank you for your interest in contributing to Vic-Torch! This document provides guidelines for contributing to the project.

## Getting Started

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MASSIVEMAGNETICS/Vic-Torch.git
   cd Vic-Torch
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, readable code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes:**
   ```bash
   pytest tests/
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: brief description of your changes"
   ```

### Commit Message Format

Use conventional commit format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `refactor:` - Code refactoring

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add comments for complex logic

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use defensive tests with `pytest.skip` for optional dependencies

## Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventional format
- [ ] PR description clearly explains the changes
- [ ] No unnecessary files committed (check .gitignore)

## Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant code snippets or error messages

## Questions?

Feel free to open an issue for questions or discussions about contributing.

Thank you for helping make Vic-Torch better! 🚀
