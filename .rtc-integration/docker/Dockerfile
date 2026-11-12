# syntax=docker/dockerfile:1.6
#
# The rtc-integration chain in one image: OpenModelica (omc), Rumoca and CasADi of their
# rtc-integration branches, built from source on Ubuntu 24.04.
#
# The branches are local, so the sources come from exports of the pinned commits (pins), made by
# prepare_context.py into context/. Each tool is built in its own stage, in one RUN that unpacks
# the sources, builds, installs to /opt/<tool> and deletes sources, build tree and download caches
# again: a stage keeps its installation only, and the final stage copies just /opt/<tool>.
#
# usage:
#   python3 prepare_context.py
#   docker build --pull -t rtc-integration -f Dockerfile .
#   docker builder prune -f     # the build cache keeps the stages' layers

ARG UBUNTU=ubuntu:24.04
ARG JOBS=8

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
RUN --mount=type=bind,source=context/openmodelica.tar,target=/ctx/openmodelica.tar \
    set -eux; \
    mkdir -p /tmp/om; tar -xf /ctx/openmodelica.tar -C /tmp/om; cd /tmp/om; \
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
RUN --mount=type=bind,source=context/rumoca.tar,target=/ctx/rumoca.tar \
    set -eux; \
    apt-get update; apt-get install -y --no-install-recommends liblzma-dev libudev-dev; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /tmp/rumoca; tar -xf /ctx/rumoca.tar -C /tmp/rumoca; cd /tmp/rumoca; \
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
RUN --mount=type=bind,source=context/swig.tar,target=/ctx/swig.tar \
    set -eux; \
    apt-get update; apt-get install -y --no-install-recommends libpcre2-dev; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /tmp/swig; tar -xf /ctx/swig.tar -C /tmp/swig; cd /tmp/swig; \
    ./autogen.sh; ./configure --prefix=/opt/swig --without-alllang --with-python3; \
    make -j "${JOBS}"; make install; \
    cd /; rm -rf /tmp/swig; \
    /opt/swig/bin/swig -version

# ---------------------------------------------------------------------------- CasADi
# .rtc-integration/build.sh of the CasADi branch, for the system Python 3.12: FMI 3 import
# (fmi-ls-dae), the ONNX translator, the lacemodelica ModelicaParser plugin (lacemodelica is
# fetched from GitHub at the commit the build script pins) and the solvers of the rtc-tools examples. The external
# projects it downloads and builds (IPOPT, MUMPS, HiGHS, Bonmin, protobuf, ONNX, ...) leave with
# the build tree; the installation is the Python package casadi and its libraries.
FROM build-tools AS casadi
ARG JOBS
COPY --from=swig /opt/swig /opt/swig
RUN --mount=type=bind,source=context/casadi.tar,target=/ctx/casadi.tar \
    set -eux; \
    apt-get update; apt-get install -y --no-install-recommends python3-dev python3-numpy libpcre2-8-0 wget; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /tmp/casadi; tar -xf /ctx/casadi.tar -C /tmp/casadi; cd /tmp/casadi; \
    PYTHON=/usr/bin/python3 SWIG=/opt/swig/bin/swig JOBS="${JOBS}" ./.rtc-integration/build.sh; \
    mkdir -p /opt; cp -r install /opt/casadi; \
    cd /; rm -rf /tmp/casadi; \
    PYTHONPATH=/opt/casadi python3 -c "import casadi; print(casadi.__version__)"

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

# pymoca and rtc-tools-channel-flow are installed and their sources removed; rtc-tools stays in
# /opt/rtc-tools (editable), as its tests and examples are what the image runs. The exports have
# no git history, so the hatch-vcs versions are given.
RUN --mount=type=bind,source=context/pymoca.tar,target=/ctx/pymoca.tar \
    --mount=type=bind,source=context/channel-flow.tar,target=/ctx/channel-flow.tar \
    --mount=type=bind,source=context/rtc-tools.tar,target=/ctx/rtc-tools.tar \
    set -eux; \
    mkdir -p /tmp/src/pymoca /tmp/src/channel-flow /opt/rtc-tools; \
    tar -xf /ctx/pymoca.tar -C /tmp/src/pymoca; \
    tar -xf /ctx/channel-flow.tar -C /tmp/src/channel-flow; \
    tar -xf /ctx/rtc-tools.tar -C /opt/rtc-tools; \
    pip install --no-cache-dir --no-deps /tmp/src/pymoca; \
    SETUPTOOLS_SCM_PRETEND_VERSION=1.5.0+rtc.integration \
      pip install --no-cache-dir --no-deps /tmp/src/channel-flow; \
    SETUPTOOLS_SCM_PRETEND_VERSION=2.9.0+rtc.integration \
      pip install --no-cache-dir --no-deps -e /opt/rtc-tools; \
    rm -rf /tmp/src; \
    python -c "import casadi, pymoca, rtctools, rtctools_channel_flow; print(casadi.__file__, pymoca.__version__, rtctools.__version__)"

WORKDIR /opt/rtc-tools
# Every tool the routes need is there (omc, rumoca, the FMI 3 headers, CasADi with ONNX import,
# fmi-ls-dae and the lacemodelica plugin, pymoca with Base Modelica)
RUN pytest -q tests/routes/test_routes.py::test_environment tests/internal/test_model_routes.py
CMD ["pytest", "-n", "3", "--dist", "loadgroup", "tests/routes"]
