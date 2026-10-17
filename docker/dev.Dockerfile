FROM fneq-remill
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends gdb && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends libclang-rt-21-dev && rm -rf /var/lib/apt/lists/*
