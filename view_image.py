import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image

parser = argparse.ArgumentParser(description='view image by name')
parser.add_argument("image", type=str, default="file", help='a name of the imag')
args = parser.parse_args()

#img = mpimg.imread(args.image)
#imgplot = plt.imshow(img)
#plt.show()

img = Image.open(args.image)
img.show()
#print("Just wait for a second if using ssh -X")

