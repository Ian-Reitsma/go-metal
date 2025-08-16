package training

import (
	"math"
)

// LRScheduler defines the interface for learning rate scheduling strategies
// All schedulers must be stateless and pure functions to maintain GPU-resident principles
type LRScheduler interface {
	// GetLR returns the learning rate for the current epoch/step
	// This is a pure function - no state modifications
	GetLR(epoch int, step int, baseLR float64) float64

	// GetName returns the scheduler name for logging
	GetName() string
}

// StepLRScheduler reduces learning rate by a factor every stepSize epochs
type StepLRScheduler struct {
	StepSize int     // Epochs between LR reductions
	Gamma    float64 // Multiplicative factor of LR decay
}

// NewStepLRScheduler creates a step learning rate scheduler
func NewStepLRScheduler(stepSize int, gamma float64) *StepLRScheduler {
	if stepSize <= 0 {
		stepSize = 30 // Default: reduce every 30 epochs
	}
	if gamma <= 0 || gamma >= 1 {
		gamma = 0.1 // Default: reduce by 10x
	}
	return &StepLRScheduler{
		StepSize: stepSize,
		Gamma:    gamma,
	}
}

func (s *StepLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	// Calculate how many times to apply gamma
	times := epoch / s.StepSize
	return baseLR * math.Pow(s.Gamma, float64(times))
}

func (s *StepLRScheduler) GetName() string {
	return "StepLR"
}

// ExponentialLRScheduler decays learning rate exponentially
type ExponentialLRScheduler struct {
	Gamma float64 // Multiplicative factor of LR decay per epoch
}

// NewExponentialLRScheduler creates an exponential learning rate scheduler
func NewExponentialLRScheduler(gamma float64) *ExponentialLRScheduler {
	if gamma <= 0 || gamma >= 1 {
		gamma = 0.95 // Default: 5% reduction per epoch
	}
	return &ExponentialLRScheduler{
		Gamma: gamma,
	}
}

func (s *ExponentialLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	return baseLR * math.Pow(s.Gamma, float64(epoch))
}

func (s *ExponentialLRScheduler) GetName() string {
	return "ExponentialLR"
}

// CosineAnnealingLRScheduler implements cosine annealing schedule
type CosineAnnealingLRScheduler struct {
	TMax   int     // Maximum number of epochs
	EtaMin float64 // Minimum learning rate
}

// NewCosineAnnealingLRScheduler creates a cosine annealing scheduler
func NewCosineAnnealingLRScheduler(tMax int, etaMin float64) *CosineAnnealingLRScheduler {
	if tMax <= 0 {
		tMax = 100 // Default: 100 epochs
	}
	if etaMin < 0 {
		etaMin = 0 // Default: anneal to 0
	}
	return &CosineAnnealingLRScheduler{
		TMax:   tMax,
		EtaMin: etaMin,
	}
}

func (s *CosineAnnealingLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	if epoch >= s.TMax {
		return s.EtaMin
	}

	// Cosine annealing formula
	return s.EtaMin + (baseLR-s.EtaMin)*(1+math.Cos(math.Pi*float64(epoch)/float64(s.TMax)))/2
}

func (s *CosineAnnealingLRScheduler) GetName() string {
	return "CosineAnnealingLR"
}

// ReduceLROnPlateauScheduler reduces LR when a metric has stopped improving
// This scheduler requires state tracking, so it's handled differently
type ReduceLROnPlateauScheduler struct {
	Factor    float64 // Factor by which the learning rate will be reduced
	Patience  int     // Number of epochs with no improvement after which LR will be reduced
	Threshold float64 // Threshold for measuring the new optimum
	Mode      string  // One of "min" or "max"

	// Internal state - these are CPU-side only for scheduling decisions
	bestMetric  float64
	badEpochs   int
	currentLR   float64
	initialized bool
}

// NewReduceLROnPlateauScheduler creates a plateau-based scheduler
func NewReduceLROnPlateauScheduler(factor float64, patience int, threshold float64, mode string) *ReduceLROnPlateauScheduler {
	if factor <= 0 || factor >= 1 {
		factor = 0.1
	}
	if patience <= 0 {
		patience = 10
	}
	if threshold < 0 {
		threshold = 1e-4
	}
	if mode != "min" && mode != "max" {
		mode = "min" // Default: minimize loss
	}

	return &ReduceLROnPlateauScheduler{
		Factor:    factor,
		Patience:  patience,
		Threshold: threshold,
		Mode:      mode,
	}
}

// Step checks if LR should be reduced based on metric
// This is called once per epoch with the validation metric
func (s *ReduceLROnPlateauScheduler) Step(metric float64, currentLR float64) float64 {
	if !s.initialized {
		s.bestMetric = metric
		s.currentLR = currentLR
		s.initialized = true
		return currentLR
	}

	improved := false
	if s.Mode == "min" {
		improved = metric < s.bestMetric-s.Threshold
	} else {
		improved = metric > s.bestMetric+s.Threshold
	}

	if improved {
		s.bestMetric = metric
		s.badEpochs = 0
	} else {
		s.badEpochs++
		if s.badEpochs >= s.Patience {
			s.currentLR *= s.Factor
			s.badEpochs = 0
		}
	}

	return s.currentLR
}

func (s *ReduceLROnPlateauScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	// For plateau scheduler, we return the internally tracked LR
	// The actual reduction happens in Step() based on metrics
	if s.initialized {
		return s.currentLR
	}
	return baseLR
}

func (s *ReduceLROnPlateauScheduler) GetName() string {
	return "ReduceLROnPlateau"
}

// NoOpScheduler maintains constant learning rate (default behavior)
type NoOpScheduler struct{}

func (s *NoOpScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	return baseLR
}

func (s *NoOpScheduler) GetName() string {
	return "ConstantLR"
}

// OneCycleLRScheduler implements the 1cycle learning rate policy.
// It increases the learning rate from a lower bound to maxLR then
// anneals back to a minimum.
type OneCycleLRScheduler struct {
	MaxLR      float64
	TotalSteps int
	PctStart   float64
	Anneal     string
	startLR    float64
	minLR      float64
}

// NewOneCycleLRScheduler creates a OneCycle scheduler.
func NewOneCycleLRScheduler(maxLR float64, totalSteps int, pctStart float64) *OneCycleLRScheduler {
	if totalSteps <= 0 {
		totalSteps = 1
	}
	if pctStart <= 0 || pctStart >= 1 {
		pctStart = 0.3
	}
	startLR := maxLR / 25.0
	minLR := startLR / 1e4
	return &OneCycleLRScheduler{
		MaxLR:      maxLR,
		TotalSteps: totalSteps,
		PctStart:   pctStart,
		Anneal:     "cos",
		startLR:    startLR,
		minLR:      minLR,
	}
}

func (s *OneCycleLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	if s.TotalSteps <= 0 {
		return baseLR
	}
	t := float64(step) / float64(s.TotalSteps)
	if t <= s.PctStart {
		// warm up
		return s.startLR + (s.MaxLR-s.startLR)*t/s.PctStart
	}
	t2 := (t - s.PctStart) / (1 - s.PctStart)
	if s.Anneal == "linear" {
		return s.MaxLR - (s.MaxLR-s.minLR)*t2
	}
	return s.minLR + (s.MaxLR-s.minLR)*(1+math.Cos(math.Pi*t2))/2
}

func (s *OneCycleLRScheduler) GetName() string {
	return "OneCycleLR"
}

// PolynomialLRScheduler decays learning rate using a polynomial schedule.
type PolynomialLRScheduler struct {
	Power    float64
	MaxSteps int
	EndLR    float64
}

// NewPolynomialLRScheduler creates a polynomial scheduler.
func NewPolynomialLRScheduler(power float64, maxSteps int, endLR float64) *PolynomialLRScheduler {
	if maxSteps <= 0 {
		maxSteps = 1
	}
	if power <= 0 {
		power = 1
	}
	return &PolynomialLRScheduler{Power: power, MaxSteps: maxSteps, EndLR: endLR}
}

func (s *PolynomialLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	if step >= s.MaxSteps {
		return s.EndLR
	}
	frac := 1 - float64(step)/float64(s.MaxSteps)
	return (baseLR-s.EndLR)*math.Pow(frac, s.Power) + s.EndLR
}

func (s *PolynomialLRScheduler) GetName() string {
	return "PolynomialLR"
}

// CyclicLRScheduler implements cyclical learning rates with optional decay.
type CyclicLRScheduler struct {
	BaseLR   float64
	MaxLR    float64
	StepSize int
	Gamma    float64
}

// NewCyclicLRScheduler creates a cyclic scheduler.
func NewCyclicLRScheduler(baseLR, maxLR float64, stepSize int, gamma float64) *CyclicLRScheduler {
	if stepSize <= 0 {
		stepSize = 1
	}
	if gamma <= 0 {
		gamma = 1.0
	}
	return &CyclicLRScheduler{BaseLR: baseLR, MaxLR: maxLR, StepSize: stepSize, Gamma: gamma}
}

func (s *CyclicLRScheduler) GetLR(epoch int, step int, baseLR float64) float64 {
	cycle := math.Floor(1 + float64(step)/(2*float64(s.StepSize)))
	x := math.Abs(float64(step)/float64(s.StepSize*2) - 2*cycle + 1)
	scale := math.Pow(s.Gamma, cycle-1)
	lr := s.BaseLR + (s.MaxLR-s.BaseLR)*math.Max(0, 1-x)*scale
	return lr
}

func (s *CyclicLRScheduler) GetName() string {
	return "CyclicLR"
}
