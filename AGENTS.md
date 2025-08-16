# AGENTS

## Project Overview
Go-Metal is a deep learning library for Go that targets Apple Silicon GPUs through Metal Performance Shaders (MPS/MPSGraph). The codebase is split into Go packages that mirror major subsystems of a training stack.

## Repository Layout
- `app/` – small demo applications that exercise the library
- `async/` – asynchronous command buffers, staging pools, and dataloaders
- `cgo_bridge/` – Objective-C bridge to Metal/MPSGraph (requires Apple's clang and Objective-C ARC)
- `checkpoints/` – save and load model weights and training state
- `docs/` – generated documentation (`task docs` regenerates package `docs.md` files)
- `engine/` – tensor engine, autograd, execution helpers
- `examples/` – stand‑alone examples referenced in the docs
- `layers/` – neural network layer implementations
- `memory/` – GPU memory management utilities
- `optimizer/` – optimization algorithms (SGD, Adam, etc.)
- `training/` – training session orchestration
- `vision/` – computer vision helpers (datasets, preprocessing, dataloaders)
- `Taskfile.yml` – common build/test/docs tasks
- `enhancements.md` – roadmap and planned features

## Development Workflow
- macOS 12+ on Apple Silicon is required; Linux containers lack Metal/ARC support
- Go 1.21 or newer
- Install Xcode Command Line Tools for Apple's clang: `xcode-select --install`
- Format Go code with `gofmt -w` (or `go fmt`) before committing
- Run `go mod tidy` after modifying dependencies
- Use `go vet` for basic static checks when available
- Regenerate documentation after changing exported APIs: `task docs`
- Commit generated `docs.md` files when they change

## Testing
- Tests that exercise the Metal bindings only run on macOS with Apple's clang and Objective‑C ARC
- Run the full suite on macOS:
  ```bash
  CC=clang go test ./...
  ```
- In non-Metal environments you can still execute the pure-Go tests:
  ```bash
  CGO_ENABLED=0 go test ./checkpoints
  ```
- The `Taskfile` exposes `task test` which wraps the macOS run above
- If Metal-dependent tests cannot be executed, note the missing dependencies in your PR description

## Useful Commands
- `task build` – build the module
- `task test` – run tests with correct CGO flags (still requires macOS)
- `task docs` – regenerate package documentation
- `task clean` – remove Go build cache

Follow these guidelines for all changes in this repository.
