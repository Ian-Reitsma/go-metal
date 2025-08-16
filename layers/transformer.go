package layers

import "fmt"

// AddMultiHeadAttention adds a multi-head attention layer to the model
func (mb *ModelBuilder) AddMultiHeadAttention(numHeads, embedDim int, dropoutRate float32, useBias bool, outputProj bool, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: MultiHeadAttention,
		Name: name,
		Parameters: map[string]interface{}{
			"num_heads":    numHeads,
			"embed_dim":    embedDim,
			"dropout_rate": dropoutRate,
			"use_bias":     useBias,
			"output_proj":  outputProj,
		},
	}
	return mb.AddLayer(layer)
}

// AddLayerNorm adds a layer normalization layer
func (mb *ModelBuilder) AddLayerNorm(normalizedShape []int, epsilon float32, elementwiseAffine bool, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: LayerNorm,
		Name: name,
		Parameters: map[string]interface{}{
			"normalized_shape":   normalizedShape,
			"epsilon":            epsilon,
			"elementwise_affine": elementwiseAffine,
		},
	}
	return mb.AddLayer(layer)
}

// AddPositionalEncoding adds a positional encoding layer
func (mb *ModelBuilder) AddPositionalEncoding(maxLength, embedDim int, encodingType, name string) *ModelBuilder {
	layer := LayerSpec{
		Type: PositionalEncoding,
		Name: name,
		Parameters: map[string]interface{}{
			"max_length":    maxLength,
			"embed_dim":     embedDim,
			"encoding_type": encodingType,
		},
	}
	return mb.AddLayer(layer)
}

// AddResidualBlock adds a residual block composed of sub-layers
func (mb *ModelBuilder) AddResidualBlock(subLayers []LayerSpec, downsample *LayerSpec, name string) *ModelBuilder {
	params := map[string]interface{}{
		"sub_layers": subLayers,
	}
	if downsample != nil {
		params["downsample"] = downsample
	}
	layer := LayerSpec{
		Type:       Residual,
		Name:       name,
		Parameters: params,
	}
	return mb.AddLayer(layer)
}

// computeMultiHeadAttentionInfo computes parameter and shape info for attention layer
func (mb *ModelBuilder) computeMultiHeadAttentionInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	if len(inputShape) != 3 {
		return nil, nil, 0, fmt.Errorf("multi-head attention requires 3D input [batch, seq_len, embed_dim]")
	}

	embedDim, ok := layer.Parameters["embed_dim"].(int)
	if !ok {
		return nil, nil, 0, fmt.Errorf("missing embed_dim parameter")
	}
	if inputShape[2] != embedDim {
		return nil, nil, 0, fmt.Errorf("input embed_dim %d doesn't match specified %d", inputShape[2], embedDim)
	}
	numHeads, ok := layer.Parameters["num_heads"].(int)
	if !ok || numHeads <= 0 || embedDim%numHeads != 0 {
		return nil, nil, 0, fmt.Errorf("invalid num_heads %v", layer.Parameters["num_heads"])
	}
	useBias := true
	if b, exists := layer.Parameters["use_bias"].(bool); exists {
		useBias = b
	}
	outputProj := true
	if o, exists := layer.Parameters["output_proj"].(bool); exists {
		outputProj = o
	}

	var paramShapes [][]int
	var paramCount int64
	weightShape := []int{embedDim, embedDim}
	biasShape := []int{embedDim}
	for i := 0; i < 3; i++ {
		paramShapes = append(paramShapes, weightShape)
		paramCount += int64(embedDim * embedDim)
		if useBias {
			paramShapes = append(paramShapes, biasShape)
			paramCount += int64(embedDim)
		}
	}
	if outputProj {
		paramShapes = append(paramShapes, weightShape)
		paramCount += int64(embedDim * embedDim)
		if useBias {
			paramShapes = append(paramShapes, biasShape)
			paramCount += int64(embedDim)
		}
	}

	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, paramShapes, paramCount, nil
}

// computeLayerNormInfo computes parameter info for layer normalization
func (mb *ModelBuilder) computeLayerNormInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	normShape, ok := layer.Parameters["normalized_shape"].([]int)
	if !ok || len(normShape) == 0 {
		return nil, nil, 0, fmt.Errorf("layer norm requires normalized_shape")
	}
	if len(inputShape) < len(normShape) {
		return nil, nil, 0, fmt.Errorf("input shape rank %d less than normalized_shape %d", len(inputShape), len(normShape))
	}
	for i := 0; i < len(normShape); i++ {
		if inputShape[len(inputShape)-len(normShape)+i] != normShape[i] {
			return nil, nil, 0, fmt.Errorf("normalized_shape %v doesn't match input shape %v", normShape, inputShape)
		}
	}
	elementwiseAffine := true
	if v, exists := layer.Parameters["elementwise_affine"].(bool); exists {
		elementwiseAffine = v
	}
	var paramShapes [][]int
	var paramCount int64
	if elementwiseAffine {
		paramShapes = append(paramShapes, normShape)
		paramShapes = append(paramShapes, normShape)
		prod := 1
		for _, d := range normShape {
			prod *= d
		}
		paramCount = int64(prod * 2)
	}
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, paramShapes, paramCount, nil
}

// computePositionalEncodingInfo computes info for positional encoding layer
func (mb *ModelBuilder) computePositionalEncodingInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	if len(inputShape) != 3 {
		return nil, nil, 0, fmt.Errorf("positional encoding requires 3D input [batch, seq_len, embed_dim]")
	}
	maxLength, ok := layer.Parameters["max_length"].(int)
	if !ok {
		return nil, nil, 0, fmt.Errorf("missing max_length parameter")
	}
	embedDim := inputShape[2]
	encodingType, _ := layer.Parameters["encoding_type"].(string)
	var paramShapes [][]int
	var paramCount int64
	if encodingType == "learned" {
		paramShapes = append(paramShapes, []int{maxLength, embedDim})
		paramCount = int64(maxLength * embedDim)
	}
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, paramShapes, paramCount, nil
}

// computeResidualInfo computes info for a residual block
func (mb *ModelBuilder) computeResidualInfo(layer *LayerSpec, inputShape []int) ([]int, [][]int, int64, error) {
	subLayers, ok := layer.Parameters["sub_layers"].([]LayerSpec)
	if !ok || len(subLayers) == 0 {
		return nil, nil, 0, fmt.Errorf("residual block requires sub_layers")
	}
	currentShape := make([]int, len(inputShape))
	copy(currentShape, inputShape)
	var paramShapes [][]int
	var paramCount int64
	for i := range subLayers {
		outShape, shapes, count, err := mb.computeLayerInfo(&subLayers[i], currentShape)
		if err != nil {
			return nil, nil, 0, err
		}
		currentShape = outShape
		paramShapes = append(paramShapes, shapes...)
		paramCount += count
	}
	if down, exists := layer.Parameters["downsample"].(*LayerSpec); exists && down != nil {
		dsShape, shapes, count, err := mb.computeLayerInfo(down, inputShape)
		if err != nil {
			return nil, nil, 0, err
		}
		inputShape = dsShape
		paramShapes = append(paramShapes, shapes...)
		paramCount += count
	}
	if len(currentShape) != len(inputShape) {
		return nil, nil, 0, fmt.Errorf("residual shape mismatch")
	}
	for i := range currentShape {
		if currentShape[i] != inputShape[i] {
			return nil, nil, 0, fmt.Errorf("residual dimension mismatch at %d", i)
		}
	}
	outputShape := make([]int, len(currentShape))
	copy(outputShape, currentShape)
	return outputShape, paramShapes, paramCount, nil
}
