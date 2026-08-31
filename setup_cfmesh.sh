#!/usr/bin/env bash
set -euo pipefail

CFMESH_VERSION="${CFMESH_VERSION:-1.2.0}"
INSTALL_ROOT="${CFMESH_INSTALL_ROOT:-$HOME/.local/cfmesh}"
INSTALL_DIR="$INSTALL_ROOT/cfMesh-$CFMESH_VERSION"
CFMESH_BIN="$INSTALL_DIR/bin/generateBoundaryLayers"
ARCHIVE="cfMesh-$CFMESH_VERSION-binaries.tgz"
URL="https://sourceforge.net/projects/cfmesh/files/$CFMESH_VERSION/$ARCHIVE/download"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: this setup script installs the self-contained Linux cfMesh binaries."
    echo "Use it on the Linux server. For Windows development, use --boundary-layers none."
    exit 1
fi

if [[ -x "$CFMESH_BIN" ]]; then
    echo "cfMesh is already installed:"
    echo "  $CFMESH_BIN"
    "$CFMESH_BIN" -help >/dev/null
    exit 0
fi

mkdir -p "$INSTALL_ROOT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading cfMesh $CFMESH_VERSION..."
if command -v curl >/dev/null 2>&1; then
    curl -fL "$URL" -o "$TMP_DIR/$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$TMP_DIR/$ARCHIVE" "$URL"
else
    echo "ERROR: curl or wget is required to download cfMesh."
    exit 1
fi

echo "Extracting into $INSTALL_ROOT..."
tar -xzf "$TMP_DIR/$ARCHIVE" -C "$INSTALL_ROOT"

if [[ ! -x "$CFMESH_BIN" ]]; then
    echo "ERROR: expected executable was not found after extraction:"
    echo "  $CFMESH_BIN"
    exit 1
fi

echo "Verifying generateBoundaryLayers..."
"$CFMESH_BIN" -help >/dev/null

echo
echo "cfMesh $CFMESH_VERSION installed successfully."
echo "Pipeline executable:"
echo "  $CFMESH_BIN"
echo
echo "No OpenFOAM environment needs to be sourced for this cfMesh package."
echo "The pipeline discovers this default installation automatically."
echo "For a custom installation, export CFMESH_BIN=/path/to/generateBoundaryLayers"
