import cv2
from random import randrange as random


#Basically loading some pre-trained data from opencv which has machine learning algorithm that can detect faces
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")




#Now we need an image to detect face from
img = cv2.imread("sample.jpg")


#Changing the image from RGB (called BGR in opencv) to Greyscale (Black and White) #cvtColor - convert color
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Returns the coordinates of the face from the grey scale in the following format ---> [[x,y(of top left), width(w), height(h)]]
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)




for i in range(len(face_coordinates)):
	print(face_coordinates[i])


#Assigning the output values of "face_coordinates" to separate variables (x,y,w,h) ----> makes it easier to work with
#N:B = [0] is used infront of "face_coordinates" because it is a list and we want the first element of the list

for (x, y, w, h) in face_coordinates:
	cv2.rectangle(img, (x,y), (x+w, y+h), (random(256),random(256),random(256)), 2)  #Combining above and below code to form a for loop

#Now we need to draw a rectangle on the main image (not the greyscale one) -----> we will use the coordinates from the "face_coordinates" variable  
#Here we are drawing a rectangle on "img" using top left coordinates (x,y) and bottom right coordinates (x+w, y+h)
#The ...(255,0,0),2) represent the COLOR of rectangle (B,G,R) and ....,2) represent the thickness of the rectangle 

#cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)





#The face detected image runs due to the following code
cv2.imshow("Face Detector", img)

#Waitkey to prevent the popup closing immediately 
cv2.waitKey()





#    ---------- TASK FOR TOMORROW --------------------

""""

(i)Write down the process of how the code is working in details
(ii) Learn the behind the scenes process (actual theory of how it is working)

"""