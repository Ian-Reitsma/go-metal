package augmentation

import (
	"image"
	"image/color"
	"math"
	"math/rand"
)

// ColorJitter adjusts brightness and contrast randomly.
type ColorJitter struct {
	Brightness float64 // max change factor, e.g., 0.2 means ±20%
	Contrast   float64 // max change factor
}

// Apply performs the color jitter.
func (cj ColorJitter) Apply(img image.Image) image.Image {
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)

	bFactor := 1 + (rand.Float64()*2-1)*cj.Brightness
	cFactor := 1 + (rand.Float64()*2-1)*cj.Contrast

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			rf := float64(r) / 65535.0
			gf := float64(g) / 65535.0
			bf := float64(b) / 65535.0

			rf = (rf-0.5)*cFactor + 0.5
			gf = (gf-0.5)*cFactor + 0.5
			bf = (bf-0.5)*cFactor + 0.5

			rf = rf * bFactor
			gf = gf * bFactor
			bf = bf * bFactor

			rf = math.Max(0, math.Min(1, rf))
			gf = math.Max(0, math.Min(1, gf))
			bf = math.Max(0, math.Min(1, bf))

			dst.Set(x, y, colorRGBA(rf, gf, bf, float64(a)/65535.0))
		}
	}
	return dst
}

func colorRGBA(r, g, b, a float64) color.Color {
	return color.RGBA64{R: uint16(r * 65535), G: uint16(g * 65535), B: uint16(b * 65535), A: uint16(a * 65535)}
}
