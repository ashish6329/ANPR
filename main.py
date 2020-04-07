import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import math
import pytesseract
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100
kNearest = cv2.ml.KNearest_create()

showSteps=True
def loadKNNDataAndTrainKNN():
	allContoursWithData = []
	validContoursWithData = []
	npaClassifications = np.loadtxt("/home/lol/Desktop/project/classifications.txt", np.float32)
	npaFlattenedImages = np.loadtxt("/home/lol/Desktop/project/flattened_images.txt", np.float32)
	npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
	kNearest.setDefaultK(1)                                                             
	kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           

	return True                            
# end function
class PossiblePlate:
	
	def __init__(self):
		self.imgPlate = None
		self.imgGrayscale = None
		self.imgThresh = None
		self.rrLocationOfPlateInScene = None
		self.strChars = ""
class PossibleChar:
    def __init__(self,contour):
        self.contour = contour
        self.boundingRect = cv2.boundingRect(self.contour)
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2)+(self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth)/float(self.intBoundingRectHeight)

###################################################################################################


def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > (80) and possibleChar.intBoundingRectWidth > (2) and possibleChar.intBoundingRectHeight > (8) and  (0.25)< possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < (0.8)):
        return True
    else:
        return False


def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []
    contours = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                       
        possibleChar = PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)       
       
    return listOfPossibleChars



def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8) #top hat
    imgBlackHat = np.zeros((height, width, 1), np.uint8)#black hat

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)# grayscale + top hat
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)# grayscale+tophat-blackhat

    return imgGrayscalePlusTopHatMinusBlackHat


def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (5,5), 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    return imgGrayscale, imgThresh


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
    print("length of contours",len(contours))

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                      

        if showSteps == True:
            cv2.drawContours(imgContours, contours, i, (255.0,255.0,255.0))
    

        possibleChar = PossibleChar(contours[i])


        if checkIfPossibleChar(possibleChar):  

            intCountOfPossibleChars = intCountOfPossibleChars + 1     
            print("possibleChar contour",(intCountOfPossibleChars))      
            listOfPossibleChars.append(possibleChar)                        
   
 

    return listOfPossibleChars,imgContours


def findListOfListsOfMatchingChars(listOfPossibleChars):
            
    listOfListsOfMatchingChars = []                  

    for possibleChar in listOfPossibleChars:                        
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        

        listOfMatchingChars.append(possibleChar)               

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue

                                                
        listOfListsOfMatchingChars.append(listOfMatchingChars)      

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             
       
        break       



    return listOfListsOfMatchingChars


def findListOfMatchingChars(possibleChar, listOfChars):
          
    listOfMatchingChars = []             

    for possibleMatchingChar in listOfChars:              
        if possibleMatchingChar == possibleChar:  
            continue                                
       
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

              
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and fltChangeInWidth < MAX_CHANGE_IN_WIDTH and fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        

    return listOfMatchingChars 


def distanceBetweenChars(firstChar, secondChar):
    intx = abs(firstChar.intCenterX - secondChar.intCenterX)
    inty = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intx ** 2) + (inty ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708                      


    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)      

    return fltAngleInDeg



def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    #cv2.destroyAllWindows()

    if showSteps == True: 
        cv2.imshow("1", imgOriginalScene)
    

    imgGrayscaleScene, imgThreshScene = preprocess(imgOriginalScene)       
    if showSteps == True:
        cv2.imshow("2", imgGrayscaleScene)
        cv2.imshow("3", imgThreshScene)
  

            
    listOfPossibleCharsInScene ,contour= findPossibleCharsInScene(imgThreshScene)

    if showSteps == True:
        

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        

        cv2.drawContours(imgContours, contours, -1,(255.0,255.0,255.0))
        cv2.imshow("4", imgContours)
  
    listOfListsOfMatchingCharsInScene = findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if showSteps == True:
        

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
           

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
       

        cv2.imshow("5", imgContours)
   

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         

        if possiblePlate.imgPlate is not None:                          
            listOfPossiblePlates.append(possiblePlate)                  
       

    if showSteps == True: 
        print("\n")
        cv2.imshow("6", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2rectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2rectPoints[0]), tuple(p2rectPoints[1]), (0.0,0.0,255.0), 2)
            cv2.line(imgContours, tuple(p2rectPoints[1]), tuple(p2rectPoints[2]), (0.0,0.0,255.0), 2)
            cv2.line(imgContours, tuple(p2rectPoints[2]), tuple(p2rectPoints[3]), (0.0,0.0,255.0), 2)
            cv2.line(imgContours, tuple(p2rectPoints[3]), tuple(p2rectPoints[0]), (0.0,0.0,255.0), 2)

            cv2.imshow("7", imgContours)

            

            #cv2.imshow("8", listOfPossiblePlates[i].imgPlate)
            
        listofMatchingCropChars=detectCharsInPlates(listOfPossiblePlates)

        print("\nplate detection complete.\n")
        #cv2.waitKey(0)

        
    

    return listofMatchingCropChars


def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate()          

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        
          
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
  

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

           
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape    

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))      

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped       

    return possiblePlate

def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:         
        return listOfPossiblePlates             


    for possiblePlate in listOfPossiblePlates:         

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess(possiblePlate.imgPlate)     

        if showSteps == True:
            cv2.imshow("9", possiblePlate.imgPlate)
            cv2.imshow("10", possiblePlate.imgGrayscale)
            cv2.imshow("11", possiblePlate.imgThresh)
        

                
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if showSteps == True: 
            cv2.imshow("12", possiblePlate.imgThresh)
            text=pytesseract.image_to_string(possiblePlate.imgThresh)
            print("image using tesseract",text)
        

            
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
       
        

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, listOfPossibleCharsInPlate)

        if showSteps == True:
            print("chars found in plate number " + str(
	           intPlateCounter) + " = " + possiblePlate.strChars)
            intPlateCounter = intPlateCounter + 1
        
            
      
    if showSteps == True:
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    return listOfPossiblePlates


def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)     

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     

    for currentChar in listOfMatchingChars:                                         
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))
        cv2.rectangle(imgThreshColor, pt1, pt2, (0.0,255.0,0.0), 2)           

                
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))         
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        npaROIReized = np.float32(npaROIResized)  
                   

        npaResults=(npaROIResized)     
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIReized, k = 50)              

        strCurrentChar = str(chr(int(npaResults[0][0])))           

        strChars = strChars + strCurrentChar                        



    if showSteps == True: 
        cv2.imshow("13", imgThreshColor)


        

   

    return strChars
def main():
    
    impath1="/home/lol/Desktop/project/dataset/"
    print("Your Image is Located at",impath1)
    file_source=input("Enter the image name \n")
    impath=impath1+file_source+".jpg"
    #print(impath)

    blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN()       

    if blnKNNTrainingSuccessful == False:                               
        print("\nerror: KNN traning was not successful\n") 
        return

    img=cv2.imread(impath)
    

    imgGrayscale = detectPlatesInScene(img)
    
if __name__ == '__main__':
    main()

