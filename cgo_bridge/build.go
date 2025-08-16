//go:build darwin && cgo
// +build darwin,cgo

package cgo_bridge

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
*/
import "C"
