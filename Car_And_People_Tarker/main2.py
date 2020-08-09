import cv2
from random import randrange
# Author : Zeeshan Khalid
Author = "Zeeshan Khalid"
# Demo Image Only For Demoing The Program !!!
file = "Pedestrians Compilation.mp4"
video = cv2.VideoCapture(file)
# Car Xml File
car_file = "cars.xml"
people_file = "people.xml"
# This Is The Xml File Classifier For Cars
car_Cascade = cv2.CascadeClassifier(car_file)
# This Is The Xml File Classifier For Pedestrians
people_Cascade = cv2.CascadeClassifier(people_file)


# This Will Run Forever
# Until a Key Is Pressed
while True:
    (read_successfull, frame) = video.read()
    # This Will Show The Video If The read_sucessfull Is True
    if read_successfull:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    car_tracker = car_Cascade.detectMultiScale(grayscaled_frame)
    people_tracker = people_Cascade.detectMultiScale(grayscaled_frame)
    for (x, y, w, h) in car_tracker:
        cv2.rectangle(frame, (x+5, y+5), (x+w, y+h),
                      (255, 0, 0), 5)

    for (x, y, w, h) in people_tracker:
        cv2.rectangle(frame, (x+5, y+5), (x+w, y+h),
                      (0, 255, 0), 5)
    cv2.imshow("Car And Pedestrians Recognition By Zeeshan Khalid", frame)
    key = cv2.waitKey(1)
