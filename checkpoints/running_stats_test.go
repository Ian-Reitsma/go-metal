package checkpoints

import (
	"reflect"
	"testing"
)

// TestGenerateIntelligentRunningStats verifies running statistics generation.
func TestGenerateIntelligentRunningStats(t *testing.T) {
	layerName := "bn1"
	numFeatures := 3

	// Case 1: No existing running statistics -> defaults
	mean, variance := generateIntelligentRunningStats(layerName, numFeatures, nil)
	expectMean := []float32{0, 0, 0}
	expectVar := []float32{1, 1, 1}
	if !reflect.DeepEqual(mean, expectMean) {
		t.Fatalf("expected mean %v, got %v", expectMean, mean)
	}
	if !reflect.DeepEqual(variance, expectVar) {
		t.Fatalf("expected var %v, got %v", expectVar, variance)
	}

	// Case 2: Existing statistics with correct length -> copy values
	stats := map[string][]float32{
		"running_mean": {1.0, 2.0, 3.0},
		"running_var":  {0.5, 0.6, 0.7},
	}
	mean, variance = generateIntelligentRunningStats(layerName, numFeatures, stats)
	expectMean = []float32{1.0, 2.0, 3.0}
	expectVar = []float32{0.5, 0.6, 0.7}
	if !reflect.DeepEqual(mean, expectMean) {
		t.Fatalf("expected mean %v, got %v", expectMean, mean)
	}
	if !reflect.DeepEqual(variance, expectVar) {
		t.Fatalf("expected var %v, got %v", expectVar, variance)
	}

	// Case 3: Existing statistics with wrong length -> defaults
	stats = map[string][]float32{
		"running_mean": {1.0},
		"running_var":  {0.5},
	}
	mean, variance = generateIntelligentRunningStats(layerName, numFeatures, stats)
	expectMean = []float32{0, 0, 0}
	expectVar = []float32{1, 1, 1}
	if !reflect.DeepEqual(mean, expectMean) {
		t.Fatalf("expected mean %v, got %v", expectMean, mean)
	}
	if !reflect.DeepEqual(variance, expectVar) {
		t.Fatalf("expected var %v, got %v", expectVar, variance)
	}
}
