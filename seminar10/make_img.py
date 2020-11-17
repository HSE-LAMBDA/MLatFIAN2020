import numpy as np

def sigmoid(x):
  return np.where(
      x > 15,
      1.,
      np.where(
          x < -15,
          0.,
          1 / (1 + np.exp(-x))
      )
  )

def soft_or(x, y, sharpness=10.):
  return sigmoid(sharpness * (x + y - 0.5))

def add_circle(img, x, y, r, sharpness=5.):
  assert img.ndim == 2
  img[:] = soft_or(
    img, np.fromfunction(
      lambda xx, yy: sigmoid(sharpness * (r**2 - ((xx - x)**2 + (yy - y)**2))),
      shape=img.shape
    )
  )

def add_square(img, x, y, r, sharpness=5.):
  assert img.ndim == 2
  img[:] = soft_or(
    img, np.fromfunction(
      lambda xx, yy: sigmoid(sharpness * (r - np.abs(xx - x))) * sigmoid(sharpness * (r - np.abs(yy - y))),
      shape=img.shape
    )
  )

img = np.zeros(shape=(300, 400), dtype=float)

np.random.seed(12358)

for _ in range(8):
  add_square(
      img,
      np.random.uniform(50, 250),
      np.random.uniform(50, 350),
      np.random.uniform(5, 25)
  )

for _ in range(8):
  add_circle(
      img,
      np.random.uniform(50, 250),
      np.random.uniform(50, 350),
      np.random.uniform(5, 25)
  )

np.save("img.npy", img)
