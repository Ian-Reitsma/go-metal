package augmentation

import (
	"image"
	"image/color"
	"testing"
)

func dummyImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 0, 255})
		}
	}
	return img
}

func TestPipeline(t *testing.T) {
	img := dummyImage(100, 100)
	p := NewPipeline(RandomCrop{Width: 50, Height: 50}, HorizontalFlip{Probability: 1.0})
	out := p.Apply(img)
	if out.Bounds().Dx() != 50 || out.Bounds().Dy() != 50 {
		t.Fatalf("unexpected output size: %v", out.Bounds())
	}
}

func TestMixUpCutMix(t *testing.T) {
	img1 := dummyImage(32, 32)
	img2 := dummyImage(32, 32)

	mix := MixUp(img1, img2, 1.0)
	if mix.Bounds() != img1.Bounds() {
		t.Fatalf("mixup size mismatch")
	}

	cut := CutMix(img1, img2, 1.0)
	if cut.Bounds() != img1.Bounds() {
		t.Fatalf("cutmix size mismatch")
	}
}
