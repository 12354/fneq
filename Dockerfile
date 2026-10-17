# fneq toolchain: LLVM 21 + remill + Alive2 + compilers + Python analysis stack.
# All external versions are pinned via the ARGs below.

ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS base
ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_MAJOR=21
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget gnupg lsb-release software-properties-common \
        git cmake ninja-build build-essential python3 python3-venv python3-pip \
        re2c pkg-config zlib1g-dev libzstd-dev libedit-dev libxml2-dev \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key > /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_MAJOR} main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        llvm-${LLVM_MAJOR} llvm-${LLVM_MAJOR}-dev llvm-${LLVM_MAJOR}-tools \
        clang-${LLVM_MAJOR} libclang-${LLVM_MAJOR}-dev lld-${LLVM_MAJOR} \
    && rm -rf /var/lib/apt/lists/*
ENV LLVM_MAJOR=${LLVM_MAJOR} \
    PATH=/usr/lib/llvm-${LLVM_MAJOR}/bin:${PATH}

# ---------------------------------------------------------------- Z3 (pinned)
FROM base AS z3
ARG Z3_VERSION=4.15.3
RUN git clone --depth 1 --branch z3-${Z3_VERSION} https://github.com/Z3Prover/z3 /src/z3 \
    && cmake -G Ninja -S /src/z3 -B /build/z3 -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/opt/z3 -DZ3_BUILD_TEST_EXECUTABLES=OFF \
    && cmake --build /build/z3 && cmake --install /build/z3 \
    && rm -rf /src /build

# ------------------------------------------------------------- Alive2 (pinned)
FROM base AS alive2
ARG ALIVE2_REF=v21.0
COPY --from=z3 /opt/z3 /opt/z3
RUN git clone https://github.com/AliveToolkit/alive2 /src/alive2 \
    && git -C /src/alive2 checkout ${ALIVE2_REF} \
    && cmake -G Ninja -S /src/alive2 -B /build/alive2 -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TV=1 -DCMAKE_PREFIX_PATH="/usr/lib/llvm-${LLVM_MAJOR};/opt/z3" \
         -DZ3_INCLUDE_DIR=/opt/z3/include -DZ3_LIBRARIES=/opt/z3/lib/libz3.so \
    && cmake --build /build/alive2 --target alive-tv \
    && mkdir -p /opt/alive2/bin && cp /build/alive2/alive-tv /opt/alive2/bin/ \
    && rm -rf /build

# ------------------------------------------------------------- remill (pinned)
FROM base AS remill
ARG REMILL_REF=v6.0.1
ENV CC=clang-${LLVM_MAJOR} CXX=clang++-${LLVM_MAJOR}
RUN git clone https://github.com/lifting-bits/remill /src/remill \
    && git -C /src/remill checkout ${REMILL_REF}
WORKDIR /src/remill
RUN cmake -G Ninja -S dependencies -B dependencies/build -DUSE_EXTERNAL_LLVM=ON \
         "-DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_MAJOR}" \
    && cmake --build dependencies/build
RUN git config --global user.email build@fneq && git config --global user.name build \
    && cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
         "-DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_MAJOR};/src/remill/dependencies/install" \
         -DCMAKE_INSTALL_PREFIX=/opt/remill \
    && cmake --build build && cmake --install build

# ------------------------------------------- fneq-lift (our remill client)
FROM remill AS fneqlift
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY lifter /lifter
RUN cmake -G Ninja -S /lifter -B /build/fneq-lift -DCMAKE_BUILD_TYPE=Release \
         "-DCMAKE_PREFIX_PATH=/opt/remill;/usr/lib/llvm-${LLVM_MAJOR};/src/remill/dependencies/install" \
    && cmake --build /build/fneq-lift \
    && cp /build/fneq-lift/fneq-lift /opt/remill/bin/

# -------------------------------------------------------------- final image
FROM base AS fneq
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg --add-architecture i386 && apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ gcc-multilib g++-multilib mingw-w64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=z3 /opt/z3 /opt/z3
COPY --from=alive2 /opt/alive2 /opt/alive2
COPY --from=fneqlift /opt/remill /opt/remill
ENV PATH=/opt/remill/bin:/opt/alive2/bin:/opt/z3/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/z3/lib
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.lock
ENV PATH=/opt/venv/bin:${PATH}

# fneq itself (pure Python; the external tools above are what it orchestrates).
COPY pyproject.toml /opt/fneq/pyproject.toml
COPY fneq /opt/fneq/fneq
COPY corpus /opt/fneq/corpus
COPY tests /opt/fneq/tests
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps /opt/fneq \
    && fneq --check-tools
ENV FNEQ_BUILD_DIR=/tmp/fneq-corpus \
    FNEQ_CORPUS_DIR=/opt/fneq/corpus
WORKDIR /work
