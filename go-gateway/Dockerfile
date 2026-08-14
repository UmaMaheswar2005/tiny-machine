# go-gateway/Dockerfile
FROM golang:1.26.1-alpine AS builder

WORKDIR /app

# Copy dependency definitions
COPY go.mod go.sum* ./
RUN go mod download

# Copy source code and build binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o gateway main.go

# Minimal runner image
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/gateway .
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["./gateway"]