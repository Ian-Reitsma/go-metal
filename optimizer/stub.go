//go:build !darwin || !cgo

package optimizer

import "github.com/tsawler/go-metal/checkpoints"

// OptimizerState is a minimal placeholder for non-Metal builds.
type OptimizerState struct {
	Type       string
	Parameters map[string]interface{}
	StateData  []checkpoints.OptimizerTensor
}

// Placeholder optimizer state types for non-Metal builds.
type AdamOptimizerState struct{}
type RMSPropOptimizerState struct{}
type SGDOptimizerState struct{}
type LBFGSOptimizerState struct{}
type AdaGradOptimizerState struct{}
type AdaDeltaOptimizerState struct{}
type NadamOptimizerState struct{}
