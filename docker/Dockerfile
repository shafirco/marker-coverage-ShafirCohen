# ---- Stage 1: build ----
FROM ubuntu:22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Build (Release)
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --config Release --parallel

# ---- Stage 2: slim runtime ----
FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libopencv-core4.5 libopencv-imgcodecs4.5 libopencv-imgproc4.5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/build/marker_coverage* /usr/local/bin/marker_coverage

# ברירת מחדל: הצג עזרה אם אין ארגומנטים
ENTRYPOINT ["/usr/local/bin/marker_coverage"]
CMD ["--help"]
