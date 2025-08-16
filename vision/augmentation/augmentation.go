// Package augmentation provides simple image data augmentation transforms.
package augmentation

import (
	"image"
	"math/rand"
	"time"
)

// Transform defines an image augmentation step.
type Transform interface {
	Apply(img image.Image) image.Image
}

// Pipeline applies a sequence of transforms.
type Pipeline struct {
	transforms []Transform
}

// NewPipeline creates a new augmentation pipeline.
func NewPipeline(transforms ...Transform) *Pipeline {
	return &Pipeline{transforms: transforms}
}

// Apply runs all transforms sequentially on the image.
func (p *Pipeline) Apply(img image.Image) image.Image {
	for _, t := range p.transforms {
		img = t.Apply(img)
	}
	return img
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
