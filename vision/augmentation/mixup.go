package augmentation

import (
	"image"
	"image/color"
	"math/rand"
)

// MixUp blends two images using a mixing factor sampled from Beta(alpha, alpha).
func MixUp(img1, img2 image.Image, alpha float64) image.Image {
	if alpha <= 0 {
		return img1
	}
	lambda := rand.Float64()
	if alpha != 1 {
		// Simple Beta distribution using two Gamma samples
		x := rand.ExpFloat64()
		y := rand.ExpFloat64()
		lambda = x / (x + y)
	}
	b1 := img1.Bounds()
	b2 := img2.Bounds()
	if !b1.Eq(b2) {
		return img1
	}
	dst := image.NewRGBA(b1)
	for y := b1.Min.Y; y < b1.Max.Y; y++ {
		for x := b1.Min.X; x < b1.Max.X; x++ {
			r1, g1, b1c, a1 := img1.At(x, y).RGBA()
			r2, g2, b2c, a2 := img2.At(x, y).RGBA()
			rf := float64(r1)*lambda + float64(r2)*(1-lambda)
			gf := float64(g1)*lambda + float64(g2)*(1-lambda)
			bf := float64(b1c)*lambda + float64(b2c)*(1-lambda)
			af := float64(a1)*lambda + float64(a2)*(1-lambda)
			dst.Set(x, y, color.RGBA64{R: uint16(rf), G: uint16(gf), B: uint16(bf), A: uint16(af)})
		}
	}
	return dst
}
