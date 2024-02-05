# plot the image as the texture of a sphere

# image_path = '/home/david/Pictures/Screenshots/Screenshot from 2024-01-26 11-43-02.png'
# image_path = '/home/david/Pictures/Screenshots/Screenshot from 2024-01-26 11-49-27.png'
image_path = '/home/david/Pictures/Screenshots/Screenshot from 2024-01-26 11-47-47.png'


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from PIL import Image

import pdb


lat_res = 100
lon_res = 200

# Load the image
img = Image.open(image_path)
img = img.resize((lon_res+1, lat_res))
img = np.array(img)
img = img / 255


# Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:lat_res*1j, 0.0:2.0*pi:lon_res*1j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

# Set the aspect ratio to 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Plot the image as a texture
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=img, shade=False)

plt.show()
