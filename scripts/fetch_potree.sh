#!/usr/bin/env bash
# Fetch the third-party Potree viewer assets that lift.viz.potree serves.
#
# Downloads the Potree 1.8.2 release and unpacks its build/ and libs/
# directories into vendor/potree/. Nothing here is committed to the repo.
#
# PotreeConverter (the octree builder) is a compiled binary and is NOT
# fetched by this script -- build it from source and drop the executable at
# vendor/potree/PotreeConverter:
#
#     git clone https://github.com/potree/PotreeConverter
#     cd PotreeConverter && mkdir build && cd build
#     cmake .. && make -j
#     cp PotreeConverter <repo>/vendor/potree/PotreeConverter
#
# Without it you can still view point clouds that already have a pre-built
# octree in their output directory.

set -euo pipefail

POTREE_VERSION="${POTREE_VERSION:-1.8.2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/vendor/potree"
URL="https://github.com/potree/potree/releases/download/${POTREE_VERSION}/Potree_${POTREE_VERSION}.zip"

if [ -d "$DEST/viewer_build" ] && [ -d "$DEST/viewer_libs" ]; then
    echo "Potree viewer assets already present in $DEST"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "downloading Potree ${POTREE_VERSION}..."
curl -fsSL "$URL" -o "$tmp/potree.zip"

echo "unpacking..."
unzip -q "$tmp/potree.zip" -d "$tmp/potree"

# The release archive nests everything one level down; find build/ and libs/
# wherever they landed.
build_dir="$(find "$tmp/potree" -maxdepth 3 -type d -name build | head -1)"
libs_dir="$(find "$tmp/potree" -maxdepth 3 -type d -name libs | head -1)"

if [ -z "$build_dir" ] || [ -z "$libs_dir" ]; then
    echo "error: could not find build/ and libs/ in the Potree archive" >&2
    exit 1
fi

mkdir -p "$DEST"
rm -rf "$DEST/viewer_build" "$DEST/viewer_libs"
cp -r "$build_dir" "$DEST/viewer_build"
cp -r "$libs_dir"  "$DEST/viewer_libs"

echo "Potree viewer assets installed in $DEST"
echo "Build PotreeConverter separately (see the header of this script)."
