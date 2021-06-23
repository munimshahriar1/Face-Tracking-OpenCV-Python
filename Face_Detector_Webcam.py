import cv2
import time
from random import randrange as random

print("\n\t\t\t Welcome to the Web Cam Face detector")
time.sleep(2)
input("\n\nPress Enter to Continue\n\n")


print("Instruction: Press Q to Quit\n\n")
print("Coordinates of detected face\n\n")


#Basically loading some pre-trained data from opencv which has machine learning algorithm that can detect faces
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")




#Now we need an image to detect face from
webcam = cv2.VideoCapture(0)

while True:
	read_frame_successs, frame = webcam.read()
	grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 #Changing the image from RGB (called BGR in opencv) to Greyscale (Black and White) #cvtColor - convert color
	face_coordinates = trained_face_data.detectMultiScale(grayscale_img)    #Returns the coordinates of the face from the grey scale in the following format ---> [[x,y(of top left), width(w), height(h)]]
	

	for (x, y, w, h) in face_coordinates:                                   #Assigning the output values of "face_coordinates" to separate variables (x,y,w,h) ----> makes it easier to work with
		

#Now we need to draw a rectangle on the main image (not the greyscale one) -----> we will use the coordinates from the "face_coordinates" variable  
#Here we are drawing a rectangle on "img" using top left coordinates (x,y) and bottom right coordinates (x+w, y+h)
#The ...(255,0,0),2) represent the COLOR of rectangle (B,G,R) and ....,2) represent the thickness of the rectangle 

		cv2.rectangle(frame, (x,y), (x+w, y+h), (random(256),random(256),random(256)), 2) 

	for i in range(len(face_coordinates)):
		print(face_coordinates[i])

		#if face_coordinates == None:
		#	print("\nNo Face Detected\n")

	cv2.imshow("Face Detector", frame)
	key = cv2.waitKey(1)             #This is the buffer between one frame and another

	if key==81 or key==113:          #ASCII key for q/Q
		break




