//go:build !darwin

package checkpoints

// OptimizerTensor is a placeholder for non-darwin builds.
type OptimizerTensor struct {
	Name       string
	Size       int
	DeviceType int
}

// WeightTensor is a placeholder for saved weights.
type WeightTensor struct {
	Name  string
	Shape []int
	Data  []float32
	Layer string
	Type  string
}

// TrainingState captures basic training progress.
type TrainingState struct {
	Epoch        int
	Step         int
	LearningRate float32
	BestLoss     float32
	BestAccuracy float32
	TotalSteps   int
}

// OptimizerState captures optimizer-specific state.
type OptimizerState struct {
	Type       string
	Parameters map[string]interface{}
	StateData  []OptimizerTensor
}
