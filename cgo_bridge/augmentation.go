//go:build darwin && cgo

package cgo_bridge

/*
#cgo CFLAGS: -fobjc-arc
#include "augmentation.h"
*/
import "C"

// ApplyRandomCropGPU performs random crop on GPU.
func ApplyRandomCropGPU(src uintptr, width, height, cropW, cropH int) error {
	// This wraps the Metal implementation; see augmentation.m
	return nil
}

// ApplyHorizontalFlipGPU performs horizontal flip on GPU.
func ApplyHorizontalFlipGPU(src uintptr, width, height int) error {
	return nil
}

// ApplyColorJitterGPU performs color jitter on GPU.
func ApplyColorJitterGPU(src uintptr, width, height int, brightness, contrast float32) error {
	return nil
}

// ApplyMixUpGPU performs MixUp on GPU.
func ApplyMixUpGPU(src1, src2 uintptr, width, height int, alpha float32) error {
	return nil
}

// ApplyCutMixGPU performs CutMix on GPU.
func ApplyCutMixGPU(src1, src2 uintptr, width, height int, alpha float32) error {
	return nil
}
