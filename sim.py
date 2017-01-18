from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# im=[]
# n=glob.glob("/home/ubuntu/imtag/modi/*.png")
# n.extend("/home/ubuntu/imtag/modi/*.cms")
# n.extend("/home/ubuntu/imtag/modi/*.jpg")
# n.extend("/home/ubuntu/imtag/modi/*.jpeg")
# #make a dict 
# ima={}

# for i in range(len(im)):
# 	ima[im[i]]=i

# visited={}

#calculates Mean squared error
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 	return s 

dim =(100,100)


# for i in range(len(im)):
# 	visited[i]=i

# for i in range(len(im)):
# 	one = cv2.imread(im[i])
# 	one = cv2.resize(one, dim, interpolation = cv2.INTER_AREA)
# 	if visited[i]==i:
# 		for j in (i+1,len(im)):
# 			two=cv2.imread(im[j])
# 			two=cv2.resize(two, dim, interpolation = cv2.INTER_AREA)
# 			s=compare_ssim(one,two)
# 			if s>=0.9 :



original = cv2.imread("copy1.cms")
contrast = cv2.imread("copy2.png")
shopped = cv2.imread("copy1.cms")


original1 = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
contrast1 = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
shopped1 = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
 
# convert the images to grayscale
original = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast1, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped1, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
 
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 
# show the figure
plt.show()
 
# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, contrast, "Original vs. Contrast")
compare_images(original, shopped, "Original vs. Photoshopped")