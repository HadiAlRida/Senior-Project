
---

# Face Recognition Script

This repository contains a script for training a face recognition model, validating it, and testing it with unknown images. The script utilizes the `face_recognition` and `Pillow` libraries.

## Directory Structure

Ensure your directory structure looks like this:

```
.
├── training
│   ├── person1
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── person2
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
├── validation
│   ├── image1.jpg
│   └── image2.jpg
├── test
│   └── unknown.jpg
├── output
│   └── encodings.pkl
└── script.py
```

- **`training/`**: Contains subdirectories for each person, with their images inside.
- **`validation/`**: Contains images for validating the trained model.
- **`test/`**: Contains the image for testing.
- **`output/`**: Will contain the `encodings.pkl` file after training.
- **`script.py`**: The script containing the provided code.

## Prerequisites

Ensure you have the necessary Python packages installed:

```bash
pip install face_recognition pillow
```

## Usage

### Training the Model

To train the model with images in the `training` directory, run:

```bash
python script.py --train -m hog
```

This will create the `encodings.pkl` file in the `output` directory using the HOG model. If you want to use the CNN model (requires GPU), replace `hog` with `cnn`:

```bash
python script.py --train -m cnn
```

### Validating the Model

To validate the trained model using the images in the `validation` directory, run:

```bash
python script.py --validate -m hog
```

Or with the CNN model:

```bash
python script.py --validate -m cnn
```

### Testing with an Unknown Image

To test the model with an unknown image, ensure you have an image file (e.g., `unknown.jpg`) in the `test` directory and run:

```bash
python script.py --test -f test/unknown.jpg -m hog
```

Or with the CNN model:

```bash
python script.py --test -f test/unknown.jpg -m cnn
```

## Explanation of Script Arguments

- `--train`: Trains the model with images in the `training` directory.
- `--validate`: Validates the model with images in the `validation` directory.
- `--test`: Tests the model with an unknown image specified by the `-f` argument.
- `-m`: Specifies the model to use, either `hog` (for CPU) or `cnn` (for GPU).
- `-f`: Specifies the path to the image with an unknown face for testing.

## Summary of the Workflow

1. **Training**: Build face encodings from labeled images in the `training` directory.
2. **Validation**: Verify that the trained model correctly recognizes faces in the `validation` directory.
3. **Testing**: Use the trained model to recognize faces in an unknown image.

By following these steps, you should be able to recognize faces in images using the provided script.

---
