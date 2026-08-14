# rust-ingestor/Dockerfile
FROM rust:latest AS builder

WORKDIR /usr/src/app

# Install build dependencies for PDF/Docx processing libraries
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build --release

# Final runtime environment
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates openssl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/app/target/release/rust-ingestor .

CMD ["./rust-ingestor"]