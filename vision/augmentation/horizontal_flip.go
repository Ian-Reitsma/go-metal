package augmentation

import (
	"image"
	"math/rand"
)

// HorizontalFlip flips the image horizontally with given probability.
type HorizontalFlip struct {
	Probability float64
}

// Apply performs the flip.
func (hf HorizontalFlip) Apply(img image.Image) image.Image {
	if rand.Float64() >= hf.Probability {
		return img
	}
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			dst.Set(bounds.Max.X-(x-bounds.Min.X)-1, y, img.At(x, y))
		}
	}
	return dst
}
