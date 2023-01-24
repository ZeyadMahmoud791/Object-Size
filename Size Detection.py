#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import numpy as np
import cv2


# In[2]:


#Function to find contours of objects
def getContours(img,minArea=1000):
    #Changing image to grey scale
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #applying gaussian blur
    Blur = cv2.GaussianBlur(grayImg,(5,5),1)
    #Canny edge detection
    cannyImg = cv2.Canny(Blur,100,100)
    #applying dilation and erosion to obtain better edges
    #this step isn't necessary 
    kernel = np.ones((5,5))
    dilation = cv2.dilate(cannyImg,kernel,iterations=3)
    imgEdges = cv2.erode(dilation,kernel,iterations=2)
    #Showing final canny image
    cv2.imshow('Canny',imgEdges)
    #obtaining contours from canny
    #use cv.CHAIN_APPROX_SIMPLE as to not store all points of an edge
    contours,hiearchy = cv2.findContours(imgEdges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    finalCountours = []
    for i in contours:
        #obtaining area of a contour
        area = cv2.contourArea(i)
        if area > minArea:
            #finding perimeters of objects
            #setting true for closed objects
            peri = cv2.arcLength(i,True)
            #approx of object curves
            #(contour,epsilon,closed)
            approx = cv2.approxPolyDP(i,0.01*peri,True)
            #returns 4 points of bounding rectangle that will be used
            #to draw the shape
            shape = cv2.boundingRect(approx)
            #info of every object found
            finalCountours.append([len(approx),area,approx,shape,i])
    #sorting objects according to their sizes
    finalCountours = sorted(finalCountours,key = lambda x:x[1] ,reverse= True)
    
    return finalCountours


# In[3]:


#function to reorder points of object to one constant sequence
def reorder(pts):
    npts = np.zeros_like(pts)
    #reshaping to remove redundant data
    pts = pts.reshape((4,2))
    #sum of (x,y) of each point
    add = pts.sum(1)
    #smallest point is (0,0)/top left
    npts[0] = pts[np.argmin(add)]
    #largest point is (1,1)/bottom right
    npts[3] = pts[np.argmax(add)]
    #difference between (x,y)
    diff = np.diff(pts,axis=1)
    #point (0,1)/top right
    npts[1]= pts[np.argmin(diff)]
    #point (1,0)/bottom left
    npts[2] = pts[np.argmax(diff)]
    return npts


# In[4]:


#function to crop A4 paper and give topdown view of it
def warpImg (img,pts,w,h,pad=20):
    #corner points after reordering
    pts =reorder(pts)
    #original points/coordinates of A4 paper
    #converting from uint8 to float32
    pts = np.float32(pts)
    #new coordinates of points/A4 paper corners as to fill whole screen
    npts = np.float32([[0,0],[w,0],[0,h],[w,h]])
    #obtaining top down view of our A4 paper
    #transformation matrix
    matrix = cv2.getPerspectiveTransform(pts,npts)
    #actual warped image
    A4Img = cv2.warpPerspective(img,matrix,(w,h))
    #padding to remove edges that are not part of the A4 paper
    A4Img = A4Img[pad:A4Img.shape[0]-pad,pad:A4Img.shape[1]-pad]
    return A4Img


# In[5]:


def dist(pts1,pts2):
    #using euclidean distance to find distance of our objects
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5


# In[7]:


#Webcam flag
webcam = False
#image path
path = r"C:\Users\zeyad\Desktop\argook.jpg"
#video settings for brightness, width and length
cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
#scaling up the image
scale = 3
#A4 paper dimensions
wP = 210 *scale
hP= 297 *scale
 
while True:
    #use webcam if flag is used else use imported image
    if webcam:
        success,img = cap.read()
    else:
        img = cv2.imread(path)
    
    #obtaining the contours of original image
    conts = getContours(img, minArea=50000)
    if len(conts) != 0:
        #obtaining the biggest obj (index 0) and the approx/corners of it (index 2)
        biggest = conts[0][2]
        #making sure the A4 paper is in frame
        if biggest.size == 8:
            #warping image
            A4Img = warpImg(img, biggest, wP,hP)
            #finding objects within our A4 paper
            A4conts = getContours(A4Img, minArea=2000)

            if len(conts) != 0:
                for obj in A4conts:
                    #drawing lines on edges of objects
                    #(img,area,closed,color,thickness)
                    print(obj[2])
                    #Skip objects that don't have 4 corner points
                    if obj[2].size!= 8:
                        continue
                    cv2.polylines(A4Img,[obj[2]],True,(0,255,0),2)
                    #reorder object points
                    npts = reorder(obj[2])
                    #finding length of each side with 1 decimal placement
                    nW = round((dist(npts[0][0]//scale,npts[1][0]//scale)/10),1)
                    nH = round((dist(npts[0][0]//scale,npts[2][0]//scale)/10),1)

                    #drawing lines and adding length as text on img
                    #(image, start_point, end_point, color, thickness, line_type, shift, tipLength)
                    cv2.arrowedLine(A4Img, (npts[0][0][0], npts[0][0][1]), (npts[1][0][0], npts[1][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    cv2.arrowedLine(A4Img, (npts[0][0][0], npts[0][0][1]), (npts[2][0][0], npts[2][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    #shape
                    x, y, w, h = obj[3]
                    #(image, text, org/coord of text, font, fontScale, color, thickness)
                    cv2.putText(A4Img, '{}cm'.format(nW), (x + 30, y - 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (255, 0, 255), 2)
                    cv2.putText(A4Img, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (255, 0, 255), 2)
            cv2.imshow('A4', A4Img)
        
    #resizing to fit screen
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()



