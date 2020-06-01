import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from skimage import io, segmentation

rgba_image = PIL.Image.open('FlowchartData/Images/00000002.jpg')
rgb_image = rgba_image.convert('RGB')
print(rgb_image)
# plt.imshow(rgb_image)
# plt.show()
# plt.savefig('myfilename.png', dpi=100)

n_segments = 50
fig_width = 2.5 * n_segments

segments = segmentation.slic(rgb_image, n_segments=n_segments)

fig, ax = plt.subplots(1, n_segments)
fig.set_figwidth(fig_width)

for index in np.unique(segments):
    segment = np.where(np.expand_dims(segments, axis=-1)==index, rgb_image, [0, 0, 0])
    print(segment)
    io.imsave('./test/test' + str(index) + '.jpg', segment)
# plt.show(fig)
