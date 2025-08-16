package checkpoints

// generateIntelligentRunningStats creates reasonable running statistics for BatchNorm layers.
// It checks for existing statistics and falls back to mean=0 and var=1 defaults.
func generateIntelligentRunningStats(layerName string, numFeatures int, runningStats map[string][]float32) ([]float32, []float32) {
	runningMean := make([]float32, numFeatures)
	runningVar := make([]float32, numFeatures)

	if runningStats != nil {
		if existingMean, ok := runningStats["running_mean"]; ok && len(existingMean) == numFeatures {
			copy(runningMean, existingMean)
		} else {
			for i := range runningMean {
				runningMean[i] = 0
			}
		}
		if existingVar, ok := runningStats["running_var"]; ok && len(existingVar) == numFeatures {
			copy(runningVar, existingVar)
		} else {
			for i := range runningVar {
				runningVar[i] = 1
			}
		}
	} else {
		// No existing statistics - use defaults
		for i := 0; i < numFeatures; i++ {
			runningMean[i] = 0
			runningVar[i] = 1
		}
	}

	return runningMean, runningVar
}
