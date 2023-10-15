# Gaussian-Based Image Reconstruction

The code in 2DGOptimizer.ipynb reconstructs a grayscale image using a combination of 2D Gaussian functions. The process involves initializing a set of Gaussians over the image, then refining their parameters (e.g., position, spread, orientation) to best match the original image.

## Libraries
```python
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from IPython.display import Video, display
```
## Result
<img src="https://raw.githubusercontent.com/uynitsuj/AIMLDLplayground/main/2D%20Gaussian%20Optimization/data/gaussian_reconstruction.gif" width="700" height="700">


## 2D Gaussian Function
Defines a 2D Gaussian function with parameters for position, spread, intensity, and rotation.

```python
def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, color_intensity, rotation_angle):
```

## Image Modeling with Gaussians
Uses the 2D Gaussian function to model an image based on the provided parameters.

```python
def model_image(params, shape):
```

## Loss Function
Defines a Mean Squared Error (MSE) loss function to measure the difference between the original image and the Gaussian-based reconstruction.

```python
def mse_loss(params):
```

## Gaussian Initialization
Initializes the Gaussians' parameters by sampling the original image on a grid.

```python
num_gaussians = 2500
grid_size = int(np.sqrt(num_gaussians))
params = []
```

## Optimization
Uses the Adam optimizer to refine the Gaussians' parameters for better image reconstruction. Also employs a learning rate scheduler for better convergence.

```python
optimizer = optim.Adam([params], lr=0.5) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
```
