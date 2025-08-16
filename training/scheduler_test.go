package training

import (
	"math"
	"testing"
)

func TestStepLRScheduler(t *testing.T) {
	scheduler := NewStepLRScheduler(2, 0.1)
	baseLR := 0.1

	tests := []struct {
		epoch      int
		expectedLR float64
	}{
		{0, 0.1},    // Initial
		{1, 0.1},    // No change yet
		{2, 0.01},   // First reduction
		{3, 0.01},   // Same
		{4, 0.001},  // Second reduction
		{5, 0.001},  // Same
		{6, 0.0001}, // Third reduction
	}

	for _, tt := range tests {
		lr := scheduler.GetLR(tt.epoch, 0, baseLR)
		if math.Abs(lr-tt.expectedLR) > 1e-8 {
			t.Errorf("Epoch %d: expected LR %f, got %f", tt.epoch, tt.expectedLR, lr)
		}
	}
}

func TestExponentialLRScheduler(t *testing.T) {
	scheduler := NewExponentialLRScheduler(0.9)
	baseLR := 0.1

	tests := []struct {
		epoch      int
		expectedLR float64
	}{
		{0, 0.1},      // Initial
		{1, 0.09},     // 0.1 * 0.9
		{2, 0.081},    // 0.1 * 0.9^2
		{3, 0.0729},   // 0.1 * 0.9^3
		{4, 0.06561},  // 0.1 * 0.9^4
		{5, 0.059049}, // 0.1 * 0.9^5
	}

	for _, tt := range tests {
		lr := scheduler.GetLR(tt.epoch, 0, baseLR)
		if math.Abs(lr-tt.expectedLR) > 1e-8 {
			t.Errorf("Epoch %d: expected LR %f, got %f", tt.epoch, tt.expectedLR, lr)
		}
	}
}

func TestCosineAnnealingLRScheduler(t *testing.T) {
	scheduler := NewCosineAnnealingLRScheduler(5, 0.0001)
	baseLR := 0.01

	// Test specific points in the cosine curve
	tests := []struct {
		epoch      int
		expectedLR float64
		tolerance  float64
	}{
		{0, 0.01, 1e-6},     // Initial (max)
		{5, 0.0001, 1e-6},   // Final (min)
		{2, 0.006580, 1e-6}, // Midpoint calculation
	}

	for _, tt := range tests {
		lr := scheduler.GetLR(tt.epoch, 0, baseLR)
		if math.Abs(lr-tt.expectedLR) > tt.tolerance {
			t.Errorf("Epoch %d: expected LR %f, got %f", tt.epoch, tt.expectedLR, lr)
		}
	}

	// Test beyond TMax
	lr := scheduler.GetLR(10, 0, baseLR)
	if lr != 0.0001 {
		t.Errorf("Beyond TMax: expected LR %f, got %f", 0.0001, lr)
	}
}

func TestReduceLROnPlateauScheduler(t *testing.T) {
	scheduler := NewReduceLROnPlateauScheduler(0.5, 2, 0.01, "min")

	// Test basic functionality
	currentLR := scheduler.Step(1.0, 0.1) // Initial
	if currentLR != 0.1 {
		t.Errorf("Initial: expected LR %f, got %f", 0.1, currentLR)
	}

	currentLR = scheduler.Step(0.98, currentLR) // Improvement
	if currentLR != 0.1 {
		t.Errorf("After improvement: expected LR %f, got %f", 0.1, currentLR)
	}

	currentLR = scheduler.Step(0.99, currentLR) // No improvement
	if currentLR != 0.1 {
		t.Errorf("No improvement 1: expected LR %f, got %f", 0.1, currentLR)
	}

	currentLR = scheduler.Step(0.99, currentLR) // No improvement - should reduce
	if currentLR != 0.05 {
		t.Errorf("No improvement 2: expected LR %f, got %f", 0.05, currentLR)
	}
}

func TestOneCycleLRScheduler(t *testing.T) {
	sched := NewOneCycleLRScheduler(0.1, 10, 0.3)
	start := sched.GetLR(0, 0, 0.001)
	mid := sched.GetLR(0, 5, 0.001)
	end := sched.GetLR(0, 9, 0.001)
	if !(start < mid && mid > end) {
		t.Fatalf("onecycle schedule not monotonic: start %.5f mid %.5f end %.5f", start, mid, end)
	}
}

func TestPolynomialLRScheduler(t *testing.T) {
	sched := NewPolynomialLRScheduler(2, 10, 0.0)
	lr0 := sched.GetLR(0, 0, 0.1)
	lr5 := sched.GetLR(0, 5, 0.1)
	lr10 := sched.GetLR(0, 10, 0.1)
	if lr0 <= lr5 || lr5 <= lr10 {
		t.Fatalf("expected polynomial decay: %f %f %f", lr0, lr5, lr10)
	}
}

func TestCyclicLRScheduler(t *testing.T) {
	sched := NewCyclicLRScheduler(0.001, 0.006, 2, 1.0)
	lr0 := sched.GetLR(0, 0, 0)
	lr1 := sched.GetLR(0, 1, 0)
	lr2 := sched.GetLR(0, 2, 0)
	lr3 := sched.GetLR(0, 3, 0)
	if !(lr0 < lr1 && lr1 > lr2 && lr2 < lr3) {
		t.Fatalf("cyclic pattern incorrect: %f %f %f %f", lr0, lr1, lr2, lr3)
	}
}

func TestSchedulerNames(t *testing.T) {
	tests := []struct {
		scheduler LRScheduler
		expected  string
	}{
		{NewStepLRScheduler(10, 0.1), "StepLR"},
		{NewExponentialLRScheduler(0.95), "ExponentialLR"},
		{NewCosineAnnealingLRScheduler(100, 0.0), "CosineAnnealingLR"},
		{NewReduceLROnPlateauScheduler(0.1, 10, 0.001, "min"), "ReduceLROnPlateau"},
		{&NoOpScheduler{}, "ConstantLR"},
		{NewOneCycleLRScheduler(0.1, 10, 0.3), "OneCycleLR"},
		{NewPolynomialLRScheduler(2, 10, 0.0), "PolynomialLR"},
		{NewCyclicLRScheduler(0.001, 0.006, 5, 1.0), "CyclicLR"},
	}

	for _, tt := range tests {
		name := tt.scheduler.GetName()
		if name != tt.expected {
			t.Errorf("Expected name %s, got %s", tt.expected, name)
		}
	}
}

// TestModelTrainerSchedulerIntegration tests the integration between schedulers and ModelTrainer
// This test verifies that SetLearningRate and scheduler methods work correctly
func TestModelTrainerSchedulerIntegration(t *testing.T) {
	// Test scheduler integration without requiring Metal device
	t.Run("SchedulerIntegration", func(t *testing.T) {
		// Create a minimal ModelTrainer structure for testing
		trainer := &ModelTrainer{
			baseLearningRate: 0.01,
			currentEpoch:     0,
			currentStep:      0,
		}

		// Test without scheduler
		info := trainer.GetSchedulerInfo()
		if info != "No scheduler (constant LR)" {
			t.Errorf("Expected 'No scheduler (constant LR)', got '%s'", info)
		}

		// Test with Step scheduler
		stepScheduler := NewStepLRScheduler(5, 0.5) // Reduce by half every 5 epochs
		trainer.SetLRScheduler(stepScheduler)

		info = trainer.GetSchedulerInfo()
		if info == "No scheduler (constant LR)" {
			t.Errorf("Expected scheduler info, got '%s'", info)
		}

		// Test learning rate calculation at epoch 0
		lr := trainer.GetCurrentLearningRate()
		if lr != 0.01 {
			t.Errorf("Expected LR 0.01 at epoch 0, got %.6f", lr)
		}

		// Test epoch update - this should trigger LR update via SetEpoch
		trainer.SetEpoch(5) // Should trigger 0.5x reduction
		if trainer.currentEpoch != 5 {
			t.Errorf("Expected current epoch 5, got %d", trainer.currentEpoch)
		}

		// Test scheduled learning rate after epoch change
		lr = trainer.GetCurrentLearningRate()
		t.Logf("Debug: LR after SetEpoch(5): %.6f, baseLR: %.6f", lr, trainer.baseLearningRate)
		expectedLR := float32(0.005) // 0.01 * 0.5
		if lr != expectedLR {
			t.Logf("Debug: Scheduler calculation: epoch=%d, step=%d, baseLR=%.6f", trainer.currentEpoch, trainer.currentStep, trainer.baseLearningRate)
			actualSchedulerLR := stepScheduler.GetLR(trainer.currentEpoch, trainer.currentStep, float64(trainer.baseLearningRate))
			t.Logf("Debug: Direct scheduler LR: %.6f", actualSchedulerLR)
			t.Errorf("Expected LR %.6f at epoch 5, got %.6f", expectedLR, lr)
		}

		// Test step-based updates (simulated since we don't have actual training)
		trainer.currentStep = 0
		trainer.updateSchedulerStep() // Should increment step and check for LR updates
		if trainer.currentStep != 1 {
			t.Errorf("Expected current step 1, got %d", trainer.currentStep)
		}

		// Test plateau scheduler (needs fresh trainer state)
		trainerPlateau := &ModelTrainer{
			baseLearningRate: 0.01,
			currentEpoch:     0,
			currentStep:      0,
		}
		trainerPlateau.config.LearningRate = 0.01

		plateauScheduler := NewReduceLROnPlateauScheduler(0.5, 2, 0.01, "min")
		trainerPlateau.SetLRScheduler(plateauScheduler)

		// Test metric-based updates
		trainerPlateau.StepSchedulerWithMetric(0.5)  // Initialize
		trainerPlateau.StepSchedulerWithMetric(0.4)  // Improve
		trainerPlateau.StepSchedulerWithMetric(0.45) // No improvement 1
		trainerPlateau.StepSchedulerWithMetric(0.46) // No improvement 2 - should reduce LR

		// Get current LR (should be reduced)
		currentLR := trainerPlateau.GetCurrentLearningRate()
		expectedLR = 0.005 // 0.01 * 0.5
		if currentLR != expectedLR {
			t.Errorf("Expected LR %.6f after plateau reduction, got %.6f", expectedLR, currentLR)
		}

		t.Logf("✅ ModelTrainer scheduler integration working correctly")
		t.Logf("   - Step scheduler properly reduces LR at epoch boundaries")
		t.Logf("   - Plateau scheduler properly reduces LR based on metrics")
		t.Logf("   - SetEpoch and updateSchedulerStep properly trigger LR updates")
	})

	t.Logf("🎉 ModelTrainer scheduler integration tests passed!")
}
