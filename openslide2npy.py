import openslide
s=openslide.OpenSlide('OS-1.ndpi')
level_counts=s.level_count
print(level_counts)

[m,n]=s.dimensions
print(m,n)

[m1,n1]=s.level_dimensions[1]
print(m1,n1)

s_downsamples=s.level_downsamples[2]
print(s_downsamples)

best_downsamples=s.get_best_level_for_downsample(5.0)
print(best_downsamples)

import numpy as np
tile=np.array(s.read_region((0,0),6,(1528,3432)))
print(tile.shape)

import matplotlib.pyplot as plt
plt.imshow(tile[:,:,0])
plt.show()
