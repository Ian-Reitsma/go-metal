package augmentation

import (
	"image"
	"image/draw"
	"math/rand"
)

// CutMix replaces a random rectangle of img1 with the same region from img2.
func CutMix(img1, img2 image.Image, alpha float64) image.Image {
	if alpha <= 0 {
		return img1
	}
	lambda := rand.Float64()
	if alpha != 1 {
		x := rand.ExpFloat64()
		y := rand.ExpFloat64()
		lambda = x / (x + y)
	}
	b := img1.Bounds()
	if !b.Eq(img2.Bounds()) {
		return img1
	}
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, img1, b.Min, draw.Src)

	cutW := int(float64(b.Dx()) * lambda)
	cutH := int(float64(b.Dy()) * lambda)
	x0 := rand.Intn(b.Dx() - cutW + 1)
	y0 := rand.Intn(b.Dy() - cutH + 1)
	rect := image.Rect(x0, y0, x0+cutW, y0+cutH)
	draw.Draw(dst, rect, img2, image.Point{X: rect.Min.X, Y: rect.Min.Y}, draw.Src)
	return dst
}
