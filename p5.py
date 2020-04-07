import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	
	impath="/home/lol/Desktop/project/dataset/test_019.jpg"
	outpath="/home/lol/Desktop/project/grey/testeg_009.jpg"
	outpath1="/home/lol/Desktop/project/noise_removal/tesetg_009.jpg"
	outpath2="/home/lol/Desktop/project/equal_histogram/teestg_009.jpg"
	outpath3="/home/lol/Desktop/project/morph_image/testg_009.jpg"
	outpath4="/home/lol/Desktop/project/Subtraction/testg_009.jpg"
	outpath5="/home/lol/Desktop/project/threshold/testg_009.jpg"
	cam = cv2.VideoCapture(0)
	s, im = cam.read() # captures image
	
	
	cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
	cv2.imshow('Original Image',im)
	img2 = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
	cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)
	noise_removal1 = cv2.bilateralFilter(img2,9,40,40)
	noise_removal = cv2.bilateralFilter(noise_removal1,9,20,20)
	cv2.namedWindow("Noise Removed Image",cv2.WINDOW_NORMAL)
	print(type(img2))
	print((img2))
	equal_histogram = cv2.equalizeHist(noise_removal)
	kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	morph_image=cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)
	
	cv2.namedWindow("Morphological opening",cv2.WINDOW_NORMAL)
	sub_morp_image = cv2.subtract(equal_histogram,morph_image)
	cv2.namedWindow("Subtraction image", cv2.WINDOW_NORMAL)

	cv2.imshow("Subtraction image", sub_morp_image)
	#ret,thresh_image1 = cv2.threshold(sub_morp_image,128,255,cv2.THRESH_BINARY)
	#ret,thresh_image =cv2.threshold(sub_morp_image,127,255,cv2.THRESH_BINARY)
	ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
	
	cv2.imshow("Image after Thresholding",thresh_image)


	
	
	

	

	cv2.imshow('Gray Converted Image',img2)
	cv2.imshow("Noise Removed Image",noise_removal)
	cv2.imshow("After Histogram equalisation",equal_histogram)
	cv2.imshow("Morphological opening",morph_image)
	cv2.imwrite(outpath3,morph_image)
	cv2.imwrite(outpath4,sub_morp_image)
	cv2.imwrite(outpath5,thresh_image)
	cv2.imwrite(outpath,img2)
	cv2.imwrite(outpath1,noise_removal)
	cv2.imwrite(outpath2,equal_histogram)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	plt.imshow(img2,cmap='Blues')
	plt.title("Blues Color map" )
	plt.xticks([])
	plt.yticks([])
	plt.show()
if __name__ == '__main__':
	main()