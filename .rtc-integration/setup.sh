#!/bin/bash
# Create the one Python environment of the rtc-integration branches: a venv from CPython 3.11, with the CasADi build (a .pth file), and pymoca, rtc-tools and rtc-tools-channel-flow
# of their rtc-integration worktrees (editable), and the other dependencies from PyPI.
#
# Layout (INTEGRATION, default the parent folder of this worktree):
#   rtc-tools/  pymoca/  casadi/ (built: casadi/install)  rumoca/ (built: target/release/rumoca)
#   ../OpenModelica/om-integration (built: build/install_cmake)
#   rtc-tools-channel-flow/ (its rtc-integration branch)
#
# usage: .rtc-integration/setup.sh; then source .rtc-integration/env.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INTEGRATION=${INTEGRATION:-$(cd "$HERE/../.." && pwd)}
VENV=${VENV:-$INTEGRATION/venv}
CHANNEL_FLOW=${CHANNEL_FLOW:-$INTEGRATION/rtc-tools-channel-flow}
rm -rf "$VENV"
# rtc-tools needs Python 3.11: a uv-managed CPython (uv python install 3.11, with
# UV_PYTHON_INSTALL_DIR=<integration>/python), the Python CasADi is built for
PYTHON=${PYTHON:-$INTEGRATION/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11}
virtualenv -q -p "$PYTHON" "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "numpy>=1.23.2" "scipy>=1.9.2" "defusedxml>=0.7.0" \
  "antlr4-python3-runtime==4.13.*" "netCDF4>=1.6.1" "rtc-tools-standard-library>=0.1.2" \
  "highspy>=1.13.1,<1.15" "pytest>=8.2.0" pytest-cov pytest-xdist "pre-commit>=3.3.0"
"$VENV/bin/pip" install -q --no-deps -e "$CHANNEL_FLOW"
# pymoca and rtc-tools as editable installs (rtc-tools reads its package metadata), without their
# declared dependencies, so that pip installs no casadi or pymoca wheel
"$VENV/bin/pip" install -q --no-deps -e "$INTEGRATION/pymoca" -e "$INTEGRATION/rtc-tools"
# CasADi is not a Python package build: its install folder goes on sys.path
SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$INTEGRATION/casadi/install" > "$SITE/rtc-integration.pth"
"$VENV/bin/python" -c "import casadi, pymoca, rtctools; print(casadi.__file__, pymoca.__file__, rtctools.__file__)"
