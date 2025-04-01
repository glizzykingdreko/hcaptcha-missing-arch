# hCaptcha Missing Arch Challenge Solver

![Final Result](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_7_original_with_missing_dot.png)

This project implements a solver for the hCaptcha "Missing Arch Challenge", where users are presented with a donut-shaped figure with one segment missing. The solver accurately determines the coordinates of the missing segment's midpoint, which is required for the challenge payload.

Check out the [Medium article](https://medium.com/@glizzykingdreko/daily-hcaptcha-challenges-chap-2-c6f51cc2cd40) too!

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Debug Mode](#debug-mode)
- [Example Solving Process](#example-solving-process)
- [How It Works](#how-it-works)
- [Author](#author)


## Overview

The solver uses computer vision techniques to:
1. Pre-process the challenge image
2. Identify the donut shape
3. Detect the missing arch
4. Calculate the precise coordinates of the missing segment's midpoint

## Features

- Robust image preprocessing with grayscale conversion and Gaussian blur
- Edge detection using the Canny algorithm
- Circle detection using Hough Circle Transform
- Intelligent arc analysis and merging
- Debug visualization capabilities
- Cartesian plane projection for verification
- Precise coordinate calculation

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/glizzykingdreko/hcaptcha-missing-arch.git
cd hcaptcha-missing-arch
```

2. Install the required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

```python
from main import DonutChallengeSolver

# Initialize the solver (enable debug mode to save intermediate images)
solver = DonutChallengeSolver(debug=True)

# Read your challenge image
with open("path/to/challenge.png", "rb") as f:
    image_bytes = f.read()

# Solve the challenge
coordinates = solver.solve(image_bytes)

if coordinates:
    print(f"Missing arch coordinates: {coordinates}")
else:
    print("Failed to solve the challenge")
```

## Debug Mode

When debug mode is enabled, the solver saves intermediate images to the `debug_output` directory, including:
- Grayscale conversion
- Blurred image
- Edge detection
- Circle detection
- Cartesian visualization
- Final result with marked coordinates

## Example Solving Process

Here's a visual demonstration of how the solver processes the challenge:

1. **Grayscale Conversion**
   ![Grayscale Image](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_1_gray.png)
   The input image is converted to grayscale to simplify processing.

2. **Gaussian Blur**
   ![Blurred Image](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_2_blurred.png)
   Gaussian blur is applied to reduce noise and smooth edges.

3. **Edge Detection**
   ![Edge Detection](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_3_edges.png)
   Canny edge detection highlights the boundaries of the donut.

4. **Circle Detection**
   ![Circle Detection](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_4_circle_detection.png)
   The Hough Circle Transform identifies the main donut shape and its boundaries.

5. **Cartesian Visualization**
   ![Cartesian Visualization](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_X_basic_cartesian.png)
   The donut is projected onto a Cartesian plane for analysis.

6. **Final Result**
   ![Final Result](https://raw.githubusercontent.com/glizzykingdreko/hcaptcha-missing-arch/main/example_solving/debug_7_original_with_missing_dot.png)
   The solver identifies the missing arch and marks its coordinates.

## How It Works

1. **Preprocessing**
   - Converts image to grayscale
   - Applies Gaussian blur to reduce noise
   - Uses Canny edge detection to highlight boundaries

2. **Donut Detection**
   - Identifies the main circle using Hough Circle Transform
   - Creates masks for inner and outer boundaries
   - Isolates the donut region

3. **Arc Analysis**
   - Groups edge points by angle
   - Merges continuous segments
   - Identifies missing arcs

4. **Coordinate Calculation**
   - Determines the largest missing arc
   - Calculates the midpoint
   - Adjusts for border inconsistencies

Inside the [images](./images/) folder you'll find some example challenges.

## Author

- GitHub: [glizzykingdreko](https://github.com/glizzykingdreko)
- Medium: [glizzykingdreko](https://medium.com/@glizzykingdreko)
- Discord: glizzykingdreko
- Email: glizzykingdreko@protonmail.com
- [Medium article](https://medium.com/@glizzykingdreko/daily-hcaptcha-challenges-chap-2-c6f51cc2cd40)

Need help passing antibots or captchas? Check out [Takion API](https://takionapi.tech)