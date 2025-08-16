//go:build !darwin || !cgo
// +build !darwin !cgo

package cgo_bridge

import (
	"fmt"
	"unsafe"
)

func errUnsupported() error { return fmt.Errorf("metal not supported on this platform") }

// DeviceType indicates where a buffer resides. These are stubbed values for
// non-Apple platforms where Metal is unavailable.
type DeviceType int

const (
	CPU DeviceType = iota
	GPU
	PersistentGPU
)

// LayerSpecC is a C-compatible representation of a layer specification used by
// the dynamic graph builder. Only the fields required by the Go side are
// included here so packages can compile without Metal support.
type LayerSpecC struct {
	LayerType        int32
	Name             [64]byte
	InputShape       [4]int32
	InputShapeLen    int32
	OutputShape      [4]int32
	OutputShapeLen   int32
	ParamInt         [8]int32
	ParamFloat       [8]float32
	ParamIntCount    int32
	ParamFloatCount  int32
	RunningMean      []float32
	RunningVar       []float32
	RunningStatsSize int32
	HasRunningStats  int32
}

// OptimizerType represents supported optimizer choices.
type OptimizerType int

const (
	SGD OptimizerType = iota
	Adam
	RMSProp
	AdaGrad
	AdaDelta
	Nadam
	LBFGS
)

// TrainingConfig is a placeholder training configuration.
type TrainingConfig struct {
	LearningRate  float32
	Beta1         float32
	Beta2         float32
	WeightDecay   float32
	Epsilon       float32
	Alpha         float32
	Momentum      float32
	Centered      bool
	OptimizerType OptimizerType
	ProblemType   int
	LossFunction  int
}

// InferenceConfig is a placeholder inference configuration.
type InferenceConfig struct {
	UseDynamicEngine       bool
	BatchNormInferenceMode bool
	UseCommandPooling      bool
	OptimizeForSingleBatch bool
	InputShape             []int32
	InputShapeLen          int32
	ProblemType            int
	LossFunction           int
	LayerSpecs             []LayerSpecC
	LayerSpecsLen          int32
}

// InferenceResult is a placeholder result structure.
type InferenceResult struct {
	Output []float32
}

// AllocateMetalBuffer is a stub that returns an error on platforms without
// Metal. It allows code depending on the cgo bridge to compile when GPU support
// is unavailable.
func AllocateMetalBuffer(device unsafe.Pointer, size int, deviceType DeviceType) (unsafe.Pointer, error) {
	return nil, errUnsupported()
}

// DeallocateMetalBuffer is a no-op stub for non-Metal builds.
func DeallocateMetalBuffer(buffer unsafe.Pointer) {}

// CreateMetalDevice returns an error on non-Metal platforms.
func CreateMetalDevice() (unsafe.Pointer, error) { return nil, errUnsupported() }

// DestroyMetalDevice is a no-op on non-Metal platforms.
func DestroyMetalDevice(device unsafe.Pointer) {}

// CreateCommandQueue returns an error on non-Metal platforms.
func CreateCommandQueue(device unsafe.Pointer) (unsafe.Pointer, error) { return nil, errUnsupported() }

// DestroyCommandQueue is a no-op on non-Metal platforms.
func DestroyCommandQueue(queue unsafe.Pointer) {}

// ReleaseCommandQueue is a no-op on non-Metal platforms.
func ReleaseCommandQueue(queue unsafe.Pointer) {}

// CreateCommandBuffer returns an error on non-Metal platforms.
func CreateCommandBuffer(commandQueue unsafe.Pointer) (unsafe.Pointer, error) {
	return nil, errUnsupported()
}

// ReleaseCommandBuffer is a no-op on non-Metal platforms.
func ReleaseCommandBuffer(commandBuffer unsafe.Pointer) {}

// CommitCommandBuffer returns an error on non-Metal platforms.
func CommitCommandBuffer(commandBuffer unsafe.Pointer) error { return errUnsupported() }

// WaitCommandBufferCompletion returns an error on non-Metal platforms.
func WaitCommandBufferCompletion(commandBuffer unsafe.Pointer) error { return errUnsupported() }

// SetupAutoreleasePool is a no-op on non-Metal platforms.
func SetupAutoreleasePool() {}

// DrainAutoreleasePool is a no-op on non-Metal platforms.
func DrainAutoreleasePool() {}

// CopyDataToStagingBuffer returns an error on non-Metal platforms.
func CopyDataToStagingBuffer(stagingBuffer unsafe.Pointer, data []byte) error {
	return errUnsupported()
}

// SetupMemoryBridgeWithConvert is a no-op stub.
func SetupMemoryBridgeWithConvert(setupFunc func(func(unsafe.Pointer, int) ([]float32, error), func(unsafe.Pointer, []float32) error, func(unsafe.Pointer, []int32) error, func(unsafe.Pointer, unsafe.Pointer, []int, int, int) error, func(unsafe.Pointer, unsafe.Pointer, int) error), getDeviceFunc func() unsafe.Pointer) {
}

// CopyTensorBufferSync returns an error on non-Metal platforms.
func CopyTensorBufferSync(srcBuffer, dstBuffer unsafe.Pointer, size int) error {
	return errUnsupported()
}

// ZeroMetalBuffer returns an error on non-Metal platforms.
func ZeroMetalBuffer(device unsafe.Pointer, buffer unsafe.Pointer, size int) error {
	return errUnsupported()
}

// ZeroMetalBufferMPSGraph returns an error on non-Metal platforms.
func ZeroMetalBufferMPSGraph(device unsafe.Pointer, buffer unsafe.Pointer, size int) error {
	return errUnsupported()
}

// CopyDataToMetalBuffer returns an error on non-Metal platforms.
func CopyDataToMetalBuffer(buffer unsafe.Pointer, data []byte) error {
	return errUnsupported()
}

// CopyFloat32ArrayToMetalBuffer returns an error on non-Metal platforms.
func CopyFloat32ArrayToMetalBuffer(buffer unsafe.Pointer, data []float32) error {
	return errUnsupported()
}

// CopyInt32ArrayToMetalBuffer returns an error on non-Metal platforms.
func CopyInt32ArrayToMetalBuffer(buffer unsafe.Pointer, data []int32) error { return errUnsupported() }

// CopyMetalBufferToFloat32Array returns an error on non-Metal platforms.
func CopyMetalBufferToFloat32Array(buffer unsafe.Pointer, numElements int) ([]float32, error) {
	return nil, errUnsupported()
}

// CopyMetalBufferToInt32Array returns an error on non-Metal platforms.
func CopyMetalBufferToInt32Array(buffer unsafe.Pointer, numElements int) ([]int32, error) {
	return nil, errUnsupported()
}

// ConvertTensorType returns an error on non-Metal platforms.
func ConvertTensorType(srcBuffer, dstBuffer unsafe.Pointer, shape []int, srcType, dstType int, device unsafe.Pointer) error {
	return errUnsupported()
}

// CopyBufferToBufferAsync returns an error on non-Metal platforms.
func CopyBufferToBufferAsync(srcBuffer, dstBuffer unsafe.Pointer, srcOffset, dstOffset, size int, commandQueue unsafe.Pointer) error {
	return errUnsupported()
}

// CopyBufferToBufferSync returns an error on non-Metal platforms.
func CopyBufferToBufferSync(srcBuffer, dstBuffer unsafe.Pointer, srcOffset, dstOffset, size int) error {
	return errUnsupported()
}

// CopyStagingToGPUBufferAsync returns an error on non-Metal platforms.
func CopyStagingToGPUBufferAsync(stagingBuffer, gpuBuffer unsafe.Pointer, stagingOffset, gpuOffset, size int, commandQueue unsafe.Pointer) error {
	return errUnsupported()
}

// WaitForBufferCopyCompletion returns an error on non-Metal platforms.
func WaitForBufferCopyCompletion(commandQueue unsafe.Pointer) error { return errUnsupported() }

// CreateTrainingEngine returns an error on non-Metal platforms.
func CreateTrainingEngine(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error) {
	return nil, errUnsupported()
}

// DestroyTrainingEngine is a no-op on non-Metal platforms.
func DestroyTrainingEngine(engine unsafe.Pointer) {}

// ExecuteTrainingStep returns an error on non-Metal platforms.
func ExecuteTrainingStep(engine, input, label unsafe.Pointer, weightBuffers []unsafe.Pointer, numWeights int, lossOut *float32) error {
	return errUnsupported()
}

// ExecuteTrainingStepDynamic returns an error on non-Metal platforms.
func ExecuteTrainingStepDynamic(engine, input, label unsafe.Pointer, weightBuffers []unsafe.Pointer, numWeights int, lossOut *float32) error {
	return errUnsupported()
}

// ExecuteTrainingStepDynamicWithGradients returns an error on non-Metal platforms.
func ExecuteTrainingStepDynamicWithGradients(engine, input, label unsafe.Pointer, weightBuffers []unsafe.Pointer, numWeights int, lossOut *float32) error {
	return errUnsupported()
}

// ExecuteTrainingStepDynamicWithGradientsPooled returns an error on non-Metal platforms.
func ExecuteTrainingStepDynamicWithGradientsPooled(engine, input, label unsafe.Pointer, weightBuffers []unsafe.Pointer, numWeights int, lossOut *float32, commandBuffer unsafe.Pointer) error {
	return errUnsupported()
}

// ExecuteTrainingStepSGDPooled returns an error on non-Metal platforms.
func ExecuteTrainingStepSGDPooled(engine, input, label unsafe.Pointer, weightBuffers []unsafe.Pointer, numWeights int, lossOut *float32, commandBuffer unsafe.Pointer) error {
	return errUnsupported()
}

// CreateInferenceEngine returns an error on non-Metal platforms.
func CreateInferenceEngine(device unsafe.Pointer, config InferenceConfig) (unsafe.Pointer, error) {
	return nil, errUnsupported()
}

// DestroyInferenceEngine is a no-op on non-Metal platforms.
func DestroyInferenceEngine(engine unsafe.Pointer) {}

// ExecuteInferenceDynamic returns an error on non-Metal platforms.
func ExecuteInferenceDynamic(engine, input unsafe.Pointer, output unsafe.Pointer, weightBuffers []unsafe.Pointer, numWeights int) error {
	return errUnsupported()
}

// ModelConfig is a placeholder for non-Metal builds.
type ModelConfig struct {
	BatchSize     int
	InputChannels int
	InputHeight   int
	InputWidth    int

	Conv1OutChannels int
	Conv1OutHeight   int
	Conv1OutWidth    int
	Conv1KernelSize  int
	Conv1Stride      int

	Conv2OutChannels int
	Conv2OutHeight   int
	Conv2OutWidth    int
	Conv2KernelSize  int
	Conv2Stride      int

	Conv3OutChannels int
	Conv3OutHeight   int
	Conv3OutWidth    int
	Conv3KernelSize  int
	Conv3Stride      int

	FC1InputSize  int
	FC1OutputSize int
	FC2OutputSize int
}

// CreateTrainingEngineDynamic returns an error on non-Metal platforms.
func CreateTrainingEngineDynamic(device unsafe.Pointer, config TrainingConfig, layerSpecs []LayerSpecC, inputShape []int) (unsafe.Pointer, error) {
	return nil, errUnsupported()
}

// CreateTrainingEngineConstantWeights returns an error on non-Metal platforms.
func CreateTrainingEngineConstantWeights(device unsafe.Pointer, config TrainingConfig) (unsafe.Pointer, error) {
	return nil, errUnsupported()
}
