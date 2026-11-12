# syntax=docker/dockerfile:1.6
#
# The rtc-integration chain in one image: OpenModelica (omc), Rumoca, CasADi (with the lacemodelica
# plugin), pymoca, rtc-tools-channel-flow and rtc-tools of their rtc-integration branches, built
# from source on Ubuntu 24.04.
#
# Everything comes from public endpoints: each repository is fetched from GitHub at the commit
# pinned below (the rtc-integration branches on github.com/yacoda), so the image builds on any
# machine and needs no build context. Each tool is built in its own stage, in one RUN that
# fetches the sources, builds, installs to /opt/<tool> and deletes sources, build tree and
# download caches again: a stage keeps its installation only, and the final stage copies just
# /opt/<tool>.
#
# usage, from anywhere:
#   docker build --pull -t rtc-integration \
#     https://github.com/yacoda/rtc-tools.git#rtc-integration:.rtc-integration/docker
#   docker run --rm rtc-integration      # runs the route matrix (tests/routes)
#   docker builder prune -f              # the build cache keeps the stages' layers

ARG UBUNTU=ubuntu:24.04
ARG JOBS=8

# The pinned commits (repository and commit per stage)
ARG OPENMODELICA_REPO=https://github.com/yacoda/OpenModelica.git
ARG OPENMODELICA_COMMIT=bc45249dd2d792b0ee0a746fefd4d849576ec8de
ARG RUMOCA_REPO=https://github.com/yacoda/rumoca.git
ARG RUMOCA_COMMIT=599cb194c7a66f3693f057f5d26916467457e173
ARG SWIG_REPO=https://github.com/jaeandersson/swig.git
ARG SWIG_COMMIT=65c2ff7fa7cf4a7895b2e9946b572b0ba1958ba0
ARG CASADI_REPO=https://github.com/yacoda/casadi.git
ARG CASADI_COMMIT=2a526203fd4c8881c5e2bcaf57685249a262d521
ARG PYMOCA_REPO=https://github.com/yacoda/pymoca.git
ARG PYMOCA_COMMIT=2e86d06e4b3f8ea743076ee847dea34f1440043b
ARG CHANNEL_FLOW_REPO=https://github.com/yacoda/rtc-tools-channel-flow.git
ARG CHANNEL_FLOW_COMMIT=23325959899ba209c07a95318256bb2f41528e11
ARG RTC_TOOLS_REPO=https://github.com/yacoda/rtc-tools.git
ARG RTC_TOOLS_COMMIT=57d564c0c5c43f463a2380844b662ddc4952961c

# ---------------------------------------------------------------------------- base
FROM ${UBUNTU} AS base
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------- build tools
# The compiler subset of the openmodelica 1.27.2 Build-Depends for noble (no Qt or GUI), and a
# rustup toolchain: OpenModelica's Rust runtime needs cargo >= 1.85, noble has 1.75.
FROM base AS build-tools
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gfortran clang cmake pkg-config autoconf automake libtool flex bison \
      default-jre-headless uuid-dev libhdf5-dev liblapack-dev libblas-dev libcurl4-gnutls-dev \
      libexpat1-dev libomp-dev libboost-dev libminizip-dev zlib1g-dev libreadline-dev \
      libncurses-dev libsqlite3-dev libffi-dev zip unzip git python3 \
 && rm -rf /var/lib/apt/lists/*
# fetch-commit <repository> <commit> <folder>: a shallow checkout of one commit
RUN printf '%s\n' '#!/bin/sh' 'set -eu' 'git init -q "$3"' \
      'git -C "$3" fetch -q --depth 1 "$1" "$2"' 'git -C "$3" checkout -q FETCH_HEAD' \
      > /usr/local/bin/fetch-commit \
 && chmod +x /usr/local/bin/fetch-commit
ENV RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | CARGO_HOME=/opt/cargo sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
 && rm -rf /opt/cargo/registry /opt/cargo/git \
 && cargo --version

# ---------------------------------------------------------------------------- OpenModelica
# Compiler and runtimes only, as .rtc-integration/build.sh of the OpenModelica branch. The
# crates go to a CARGO_HOME inside the build tree, so that they leave with it.
FROM build-tools AS omc
ARG JOBS
ARG OPENMODELICA_REPO
ARG OPENMODELICA_COMMIT
# The submodules the build uses; their .gitmodules URLs are relative, so they point at
# OpenModelica's own repositories (the integration branch changes none of them)
RUN set -eux; \
    fetch-commit "${OPENMODELICA_REPO}" "${OPENMODELICA_COMMIT}" /tmp/om; cd /tmp/om; \
    for sub in OMCompiler/3rdParty:OMCompiler-3rdParty OMCompiler/Compiler/boot/bomc:OMBootstrapping \
               OMCompiler/Examples/ModelicaTutorial:ModelicaTutorial; do \
      git submodule init "${sub%%:*}"; \
      git config "submodule.${sub%%:*}.url" "https://github.com/OpenModelica/${sub##*:}.git"; \
    done; \
    git submodule update --depth 1 --recursive --jobs "${JOBS}" \
      OMCompiler/3rdParty OMCompiler/Compiler/boot/bomc OMCompiler/Examples/ModelicaTutorial; \
    export CARGO_HOME=/tmp/om/cargo-home CARGO_NET_RETRY=10 CARGO_HTTP_TIMEOUT=300; \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/openmodelica \
      -DOM_ENABLE_GUI_CLIENTS=OFF -DOM_ENABLE_OMSIMULATOR=OFF -DOM_ENABLE_TESTSUITE=OFF \
      -DOM_QT_MAJOR_VERSION=5 -DOM_USE_CCACHE=OFF; \
    cmake --build build --parallel "${JOBS}" --target install; \
    cd /; rm -rf /tmp/om; \
    /opt/openmodelica/bin/omc --version

# ---------------------------------------------------------------------------- Rumoca
# The rumoca command line tool, as .rtc-integration/build.sh of the Rumoca branch, and the FMI 3
# headers it ships, with which rtc-tools compiles the C sources of Rumoca's FMUs. Release build
# without incremental compilation; crates and target directory leave with the build tree.
FROM build-tools AS rumoca
ARG JOBS
ARG RUMOCA_REPO
ARG RUMOCA_COMMIT
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends liblzma-dev libudev-dev; \
    rm -rf /var/lib/apt/lists/*; \
    fetch-commit "${RUMOCA_REPO}" "${RUMOCA_COMMIT}" /tmp/rumoca; cd /tmp/rumoca; \
    export CARGO_HOME=/tmp/rumoca/cargo-home CARGO_TARGET_DIR=/tmp/rumoca/target \
      CARGO_INCREMENTAL=0 CARGO_NET_RETRY=10 CARGO_HTTP_TIMEOUT=300; \
    cargo build --release --bin rumoca --jobs "${JOBS}"; \
    install -D -m 755 target/release/rumoca /opt/rumoca/bin/rumoca; \
    mkdir -p /opt/rumoca/share; \
    cp -r crates/rumoca-phase-codegen/src/templates/fmi-ls-wasm/fmi3-headers /opt/rumoca/share/fmi3-headers; \
    cd /; rm -rf /tmp/rumoca; \
    /opt/rumoca/bin/rumoca --version

# ---------------------------------------------------------------------------- SWIG
# CasADi's Python interface is generated with its own SWIG fork (jaeandersson/swig); only the
# CasADi stage uses it.
FROM build-tools AS swig
ARG JOBS
ARG SWIG_REPO
ARG SWIG_COMMIT
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends libpcre2-dev; \
    rm -rf /var/lib/apt/lists/*; \
    fetch-commit "${SWIG_REPO}" "${SWIG_COMMIT}" /tmp/swig; cd /tmp/swig; \
    ./autogen.sh; ./configure --prefix=/opt/swig --without-alllang --with-python3; \
    make -j "${JOBS}"; make install; \
    cd /; rm -rf /tmp/swig; \
    /opt/swig/bin/swig -version

# ---------------------------------------------------------------------------- CasADi
# .rtc-integration/build.sh of the CasADi branch, for the system Python 3.12: FMI 3 import
# (fmi-ls-dae), the ONNX translator, the lacemodelica ModelicaParser plugin (lacemodelica is
# fetched from GitHub at the commit the build script pins) and the solvers of the rtc-tools
# examples. The external projects it downloads and builds (IPOPT, MUMPS, HiGHS, Bonmin, protobuf, ONNX, ...) leave with
# the build tree; the installation is the Python package casadi and its libraries.
FROM build-tools AS casadi
ARG JOBS
ARG CASADI_REPO
ARG CASADI_COMMIT
COPY --from=swig /opt/swig /opt/swig
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends python3-dev python3-numpy libpcre2-8-0 wget; \
    rm -rf /var/lib/apt/lists/*; \
    fetch-commit "${CASADI_REPO}" "${CASADI_COMMIT}" /tmp/casadi; cd /tmp/casadi; \
    PYTHON=/usr/bin/python3 SWIG=/opt/swig/bin/swig JOBS="${JOBS}" ./.rtc-integration/build.sh; \
    mkdir -p /opt; cp -r install /opt/casadi; \
    cd /; rm -rf /tmp/casadi; \
    PYTHONPATH=/opt/casadi python3 -c "import casadi; print(casadi.__version__)"

# ---------------------------------------------------------------------------- Python sources
# pymoca, rtc-tools-channel-flow and rtc-tools, fetched here, where git is, for the last stage
FROM build-tools AS python-sources
ARG PYMOCA_REPO
ARG PYMOCA_COMMIT
ARG CHANNEL_FLOW_REPO
ARG CHANNEL_FLOW_COMMIT
ARG RTC_TOOLS_REPO
ARG RTC_TOOLS_COMMIT
RUN set -eux; \
    fetch-commit "${PYMOCA_REPO}" "${PYMOCA_COMMIT}" /src/pymoca; \
    fetch-commit "${CHANNEL_FLOW_REPO}" "${CHANNEL_FLOW_COMMIT}" /src/channel-flow; \
    fetch-commit "${RTC_TOOLS_REPO}" "${RTC_TOOLS_COMMIT}" /src/rtc-tools; \
    rm -rf /src/*/.git

# ---------------------------------------------------------------------------- runtime
# omc writes C code and compiles it for every FMU it exports: the image keeps a C toolchain and
# cmake, and the run-time libraries omc links.
FROM base AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gfortran cmake liblapack3 libblas3 libgomp1 libomp5 libcurl3t64-gnutls \
      libexpat1 libuuid1 libhdf5-103-1t64 libminizip1t64 zlib1g libreadline8t64 libsqlite3-0 \
      libffi8 libstdc++6 unzip liblzma5 libudev1 python3 python3-numpy libgfortran5 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=omc /opt/openmodelica /opt/openmodelica
COPY --from=rumoca /opt/rumoca /opt/rumoca
COPY --from=casadi /opt/casadi /opt/casadi
ENV PATH=/opt/openmodelica/bin:/opt/rumoca/bin:${PATH} \
    RTCTOOLS_OMC=omc RTCTOOLS_RUMOCA=rumoca RTCTOOLS_FMI3_INCLUDE=/opt/rumoca/share/fmi3-headers

# Smoke test: Base Modelica and FMUs in ODE and DAE mode (fmi-ls-dae) of a small model, by omc
# and by Rumoca (also its casadi-dae output, for rtc-tools' Rumoca loader)
RUN set -eux; mkdir /tmp/smoke; cd /tmp/smoke; \
    printf 'model M\n  parameter Real k = 2;\n  Real x(start = 1, fixed = true);\n  Real y;\nequation\n  der(x) = -k * x;\n  y = x^2;\nend M;\n' > M.mo; \
    printf 'loadFile("M.mo"); getErrorString();\nwriteFile("M.bmo", instantiateModel(M)); getErrorString();\n' > bmo.mos; \
    omc --BaseModelica --baseModelicaFormat=scalarized bmo.mos; grep -q "base 0.1.0" M.bmo; \
    printf 'loadFile("M.mo"); getErrorString();\nbuildModelFMU(M, version="3.0", fmuType="me", fileNamePrefix="M"); getErrorString();\n' > fmu.mos; \
    omc --fmiInputDerivatives fmu.mos; unzip -l M.fmu | grep -q "binaries/x86_64-linux/M.so"; \
    mv M.fmu M_ode.fmu; \
    omc --daeMode --simCodeTarget=C+Rust --fmiInputDerivatives fmu.mos; \
    unzip -l M.fmu | grep -q "binaries/x86_64-linux/M.so"; unzip -l M.fmu | grep -q fmi-ls-dae; \
    rumoca compile M.mo --model M --target base-modelica --output rumoca-bmo; \
    grep -q "base 0.1.0" rumoca-bmo/M_base.mo; \
    rumoca compile M.mo --model M --target casadi-dae --output rumoca-casadi; \
    ls rumoca-casadi/M_casadi_dae.py; \
    rumoca compile M.mo --model M --target fmi3 --output rumoca-fmu; \
    test -s rumoca-fmu/M.fmu; test -s "${RTCTOOLS_FMI3_INCLUDE}/fmi3Functions.h"; \
    PYTHONPATH=/opt/casadi python3 -c "import casadi as ca, glob, os; \
assert hasattr(ca, 'GraphBuilder'), 'no GraphBuilder'; \
assert glob.glob(os.path.dirname(ca.__file__) + '/libcasadi_graphmodel_onnx*'), 'no ONNX import'; \
print('casadi', ca.__version__)"; \
    PYTHONPATH=/opt/casadi python3 -c "import casadi as ca; \
[ca.load_nlpsol(n) for n in ('ipopt', 'bonmin')]; [ca.load_conic(n) for n in ('cbc', 'clp', 'highs')]; \
x = ca.MX.sym('x'); s = ca.nlpsol('s', 'ipopt', {'x': x, 'f': (x - 1)**2}, {'print_time': False, 'ipopt.print_level': 0}); \
assert abs(float(s(x0=0)['x']) - 1) < 1e-8; print('solver plugins and an IPOPT solve: ok')"; \
    PYTHONPATH=/opt/casadi python3 -c "import casadi as ca; \
ca.ModelicaParser('lacemodelica').parse('/tmp/smoke/M.bmo', '/tmp/smoke/M_lace'); \
g = ca.GraphBuilder('/tmp/smoke/M_lace/extra/org.lacemodelica.ls-onnx-serialization/model.onnx').create('f', {'symbolic': True}); \
print('lacemodelica plugin:', g.name_in(), g.name_out())"; \
    mkdir M_dae; cd M_dae; unzip -q ../M.fmu; cd ..; \
    PYTHONPATH=/opt/casadi python3 -c "import casadi as ca; \
dae = ca.DaeBuilder('M', '/tmp/smoke/M_dae'); \
print('fmi-ls-dae import:', dae.x(), dae.z())"; \
    cd /; rm -rf /tmp/smoke

# ---------------------------------------------------------------------------- rtc-tools
# The Python side: a venv on the system Python 3.12 with pymoca, rtc-tools-channel-flow and
# rtc-tools of their rtc-integration branches (no wheels of casadi or pymoca: CasADi comes from
# /opt/casadi through a .pth file), the Modelica Standard Library 4.0.0 for omc's exports, and
# the route tests as the default command.
FROM runtime AS rtc-integration
ARG MSL_VERSION=4.0.0
ARG MSL_SHA256=c5ebed218f20320b95aa5086c1c7de6a28bd1614e7bc14dc084523b5fa6a713c
RUN set -eux; \
    curl -sSL -o /tmp/msl.tgz \
      "https://github.com/modelica/ModelicaStandardLibrary/archive/refs/tags/v${MSL_VERSION}.tar.gz"; \
    echo "${MSL_SHA256}  /tmp/msl.tgz" | sha256sum -c; \
    tar -xzf /tmp/msl.tgz -C /tmp; mkdir -p /opt/modelica/libraries; \
    mv "/tmp/ModelicaStandardLibrary-${MSL_VERSION}/Modelica" "/opt/modelica/libraries/Modelica ${MSL_VERSION}"; \
    mv "/tmp/ModelicaStandardLibrary-${MSL_VERSION}/ModelicaServices" \
       "/opt/modelica/libraries/ModelicaServices ${MSL_VERSION}"; \
    mv "/tmp/ModelicaStandardLibrary-${MSL_VERSION}/Complex.mo" /opt/modelica/libraries/; \
    rm -rf /tmp/msl.tgz "/tmp/ModelicaStandardLibrary-${MSL_VERSION}"
ENV OPENMODELICALIBRARY=/opt/modelica/libraries

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir "numpy>=1.23.2" "scipy>=1.9.2" "defusedxml>=0.7.0" \
      "antlr4-python3-runtime==4.13.*" "netCDF4>=1.6.1" "rtc-tools-standard-library>=0.1.2" \
      "highspy>=1.13.1,<1.15" "pytest>=8.2.0" pytest-xdist \
 && echo /opt/casadi > "$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/rtc-integration.pth"
ENV PATH=/opt/venv/bin:${PATH}

# pymoca and rtc-tools-channel-flow are installed from the python-sources stage; rtc-tools is
# copied to /opt/rtc-tools (editable), as its tests and examples are what the image runs. The
# sources have no git history, so the hatch-vcs versions are given.
COPY --from=python-sources /src/rtc-tools /opt/rtc-tools
RUN --mount=type=bind,from=python-sources,source=/src,target=/src \
    set -eux; \
    cp -r /src/pymoca /src/channel-flow /tmp/; \
    pip install --no-cache-dir --no-deps /tmp/pymoca; \
    SETUPTOOLS_SCM_PRETEND_VERSION=1.5.0+rtc.integration \
      pip install --no-cache-dir --no-deps /tmp/channel-flow; \
    SETUPTOOLS_SCM_PRETEND_VERSION=2.9.0+rtc.integration \
      pip install --no-cache-dir --no-deps -e /opt/rtc-tools; \
    rm -rf /tmp/pymoca /tmp/channel-flow; \
    python -c "import casadi, pymoca, rtctools, rtctools_channel_flow; print(casadi.__file__, pymoca.__version__, rtctools.__version__)"

WORKDIR /opt/rtc-tools
# Every tool the routes need is there (omc, rumoca, the FMI 3 headers, CasADi with ONNX import,
# fmi-ls-dae and the lacemodelica plugin, pymoca with Base Modelica)
RUN pytest -q tests/routes/test_routes.py::test_environment tests/internal/test_model_routes.py
CMD ["pytest", "-n", "3", "--dist", "loadgroup", "tests/routes"]
