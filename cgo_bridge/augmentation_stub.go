//go:build !darwin || !cgo

package cgo_bridge

import "errors"

// ApplyRandomCropGPU is a stub for platforms without Metal support.
func ApplyRandomCropGPU(src uintptr, width, height, cropW, cropH int) error {
	return errors.New("metal augmentation not supported on this platform")
}

// ApplyHorizontalFlipGPU is a stub for platforms without Metal support.
func ApplyHorizontalFlipGPU(src uintptr, width, height int) error {
	return errors.New("metal augmentation not supported on this platform")
}

// ApplyColorJitterGPU is a stub for platforms without Metal support.
func ApplyColorJitterGPU(src uintptr, width, height int, brightness, contrast float32) error {
	return errors.New("metal augmentation not supported on this platform")
}

// ApplyMixUpGPU is a stub for platforms without Metal support.
func ApplyMixUpGPU(src1, src2 uintptr, width, height int, alpha float32) error {
	return errors.New("metal augmentation not supported on this platform")
}

// ApplyCutMixGPU is a stub for platforms without Metal support.
func ApplyCutMixGPU(src1, src2 uintptr, width, height int, alpha float32) error {
	return errors.New("metal augmentation not supported on this platform")
}
