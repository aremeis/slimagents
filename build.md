# Build and upload to pypi

Install required dependencies
```shell
pip install build twine
```

Clean build artifacts
```shell
rm -rf build/ dist/ *.egg-info/ **/__pycache__/ **/*.py[cod]
```

Build package
```shell
python -m build
```

Upload to testpypi (API token required, or use `~/.pypirc` to store credentials)
```shell
python -m twine upload --repository testpypi dist/*
```

Verify upload to testpypi (in a new virtual environment)
```shell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ slimagents
python -c "import slimagents; print(slimagents.__version__)"
```

Upload to pypi
```shell
python -m twine upload dist/*
```

Build, tag, release and upload to pypi
```shell
./release.sh
```