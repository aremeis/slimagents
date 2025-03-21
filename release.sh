#!/bin/bash
set -e  # Exit on error

# Extract version from pyproject.toml using a simple grep and cut approach
VERSION=$(grep "^version = " pyproject.toml | cut -d'"' -f2)

if [ -z "$VERSION" ]; then
  echo "Error: Could not extract version from pyproject.toml"
  exit 1
fi

echo "Found version: $VERSION"

# Ask for confirmation
read -p "Create release for version v$VERSION? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Release cancelled."
  exit 0
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
  echo "Warning: You have uncommitted changes."
  read -p "Do you want to commit all changes with message 'Release v$VERSION'? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add .
    git commit -m "Release v$VERSION"
  else
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Release cancelled."
      exit 0
    fi
  fi
fi

# Push to GitHub if requested
read -p "Push changes to GitHub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Pushing to GitHub..."
  git push
fi

# Check if tag already exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
  read -p "Tag v$VERSION already exists. Continue with release creation only? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Release cancelled."
    exit 0
  fi
  TAG_EXISTS=1
else
  TAG_EXISTS=0
fi

# Create tag if it doesn't exist
if [ $TAG_EXISTS -eq 0 ]; then
  echo "Creating tag v$VERSION..."
  git tag -a "v$VERSION" -m "Release version $VERSION"
  
  echo "Pushing tag to GitHub..."
  git push origin "v$VERSION"
else
  echo "Tag v$VERSION already exists, skipping tag creation."
fi

# Create GitHub release using GitHub CLI (if available)
if command -v gh &>/dev/null; then
  echo "Creating GitHub release using GitHub CLI..."
  read -p "Enter release notes (or press Enter to use default message): " RELEASE_NOTES
  if [ -z "$RELEASE_NOTES" ]; then
    RELEASE_NOTES="Release version $VERSION"
  fi
  
  gh release create "v$VERSION" --title "v$VERSION" --notes "$RELEASE_NOTES"
  echo "GitHub release created: v$VERSION"
else
  echo "GitHub CLI (gh) not found."
  echo "To create a GitHub release, install GitHub CLI or manually create it at:"
  echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:\/]\(.*\)\.git/\1/')/releases/new?tag=v$VERSION"
fi

# Check if build and upload tools are installed
if ! command -v python -m build &>/dev/null || ! command -v python -m twine &>/dev/null; then
  echo "Warning: build or twine not found. Installing required dependencies..."
  pip install build twine
fi

# Build and upload to PyPI
read -p "Clean, build, and upload package to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  # Clean build artifacts
  echo "Cleaning build artifacts..."
  rm -rf build/ dist/ *.egg-info/ **/__pycache__/ **/*.py[cod]
  
  # Build package
  echo "Building package..."
  python -m build
  
  # Ask about testpypi
  read -p "Upload to Test PyPI first? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to Test PyPI..."
    python -m twine upload --repository testpypi dist/*
    
    echo "Verify the test package with:"
    echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ slimagents"
    echo "python -c \"import slimagents; print(slimagents.__version__)\""
    
    read -p "Continue with upload to PyPI? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "PyPI upload cancelled. Release process partially completed."
      exit 0
    fi
  fi
  
  # Upload to PyPI
  echo "Uploading to PyPI..."
  python -m twine upload dist/*
fi

echo "Release process completed!" 