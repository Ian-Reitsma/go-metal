# augmentation
--
    import "."

Package augmentation provides simple image data augmentation transforms.

## Usage

#### type Pipeline

```go
type Pipeline struct {
        // contains filtered or unexported fields
}
```

Pipeline applies a sequence of image transforms.

#### func  NewPipeline

```go
func NewPipeline(transforms ...Transform) *Pipeline
```

NewPipeline creates a new augmentation pipeline.

#### type RandomCrop

```go
type RandomCrop struct {
        Width  int
        Height int
}
```

RandomCrop randomly crops an image to the specified size.

#### type HorizontalFlip

```go
type HorizontalFlip struct {
        Probability float64
}
```

HorizontalFlip flips the image horizontally with given probability.

#### type ColorJitter

```go
type ColorJitter struct {
        Brightness float64
        Contrast   float64
}
```

ColorJitter adjusts brightness and contrast randomly.

#### func  MixUp

```go
func MixUp(img1, img2 image.Image, alpha float64) image.Image
```

MixUp blends two images using a mixing factor sampled from Beta(alpha, alpha).

#### func  CutMix

```go
func CutMix(img1, img2 image.Image, alpha float64) image.Image
```

CutMix replaces a random rectangle of img1 with the same region from img2.

