## Steps to Build and Upload the Package to PyPI:

1. **Install necessary tools**:

```bash
   python -m pip install --upgrade build twine
```

2. **Build the package**:
   Run this command in the root directory where pyproject.toml is located:

```bash
python -m build
```

This will generate the dist/ directory containing the .tar.gz and .whl files.

3. Upload the package to PyPI:

Ensure you have an account on PyPI. Then use twine to upload your package:

```bash
python -m twine upload dist/*
```

You will be prompted to enter your PyPI username and password.

Once uploaded, your package will be available on PyPI, and users can install it using:

```bash
pip install prot2token
```
