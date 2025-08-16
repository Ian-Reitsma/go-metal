package augmentation

import (
	"image"
	"image/draw"
	"math/rand"
)

// RandomCrop randomly crops an image to the specified size.
type RandomCrop struct {
	Width  int
	Height int
}

// Apply performs the random crop.
func (rc RandomCrop) Apply(img image.Image) image.Image {
	bounds := img.Bounds()
	if rc.Width <= 0 || rc.Height <= 0 || rc.Width > bounds.Dx() || rc.Height > bounds.Dy() {
		return img
	}
	x0 := rand.Intn(bounds.Dx() - rc.Width + 1)
	y0 := rand.Intn(bounds.Dy() - rc.Height + 1)
	rect := image.Rect(0, 0, rc.Width, rc.Height)
	dst := image.NewRGBA(rect)
	draw.Draw(dst, rect, img, image.Point{X: x0, Y: y0}, draw.Src)
	return dst
}
