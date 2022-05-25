import csv
import os
import pytesseract
import numpy as np
import datetime
from simple_facerec import SimpleFacerec
import cv2

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {"AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar",
          "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu", "DL": "Delhi", "GA": "Goa",
          "GJ": "Gujarat",
          "HR": "Haryana", "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
          "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya",
          "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odissa", "PY": "Pondicherry", "PN": "Punjab", "RJ": "Rajasthan",
          "SK": "Sikkim", "TN": "TamilNadu", "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal",
          "CG": "Chhattisgarh", "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"}

# Load Camera
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('C:/Users/gopal/Downloads/source-code-face-recognition/source code/video.MOV')
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    return frame


def extract_num(img_name):
    img = img_name
    # img = cv2.imread(img_name) ## Reading Image
    # Converting into Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecting plate
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in nplate:
        # Crop a portion of plate
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        plate = img[y + a:y + h - a, x + b:x + w - b + 25, :]
        # make image more darker to identify the LPR
        ## iMAGE PROCESSING
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # Feed Image to OCR engine
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        print(read)
        if((len(read) >= 5 and len(read) <= 10) and not os.path.isfile(str(read)+'.jpg')):
            data=[str(read),str(datetime.datetime.now())]
            writer.writerow(data)
        stat = read[0:2]
        try:
            # Fetch the State information
            print('Car Belongs to', states[stat])
        except FileNotFoundError:
            continue
        except:
            print('State not recognised!!')

        print(read)
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.imshow('PLate',plate)
        # Save & display result image
        if ((len(read) >= 5 and len(read) <= 10) and not os.path.isfile(str(read)+'.jpg')):
            cv2.imwrite('plate' + read + '.jpg', plate)
        else:
            cv2.imwrite('plateUNKNOWN.jpg',plate)

    # cv2.imshow("Result", img)
    # cv2.imwrite('result.jpg',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def detectface(framecap = None):
    i=0
    f = open('records.csv', 'w')
    writer = csv.writer(f)
    if(framecap is None):
        ret, frame = cap.read()
    else:
        ret, frame=framecap
    # to show time and date
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Width: ' + str(cap.get(3)) + ' Height:' + str(cap.get(4))
        datet = str(datetime.datetime.now())
        """frame = cv2.putText(frame, text, (10, 100), font, 1,
                            (0, 255, 255), 2, cv2.LINE_AA)"""
        frame = cv2.putText(frame, datet, (10, 25), font, 0.5,
                            (0, 255, 255), 2, cv2.LINE_AA)
    # Detect Faces
    face_locations, face_names, face_times = sfr.detect_known_faces(frame)
    for face_loc, name, times in zip(face_locations, face_names, face_times):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        # cv2.putText(frame, times, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        print(times)
        writer.writerow(times)
        img = frame[x1:y1, x2:y2]
        # cv2.imwrite(os.path.join("C:/Users/gopal/Downloads/source-code-face-recognition/source code/pics" ,'Frame' + str(i) + '.jpg'), img)
        i += 1
    #detect(frame)
    cv2.imshow("Frame", frame)
    extract_num(frame)
    return frame
    f.close()


#detectface()
cap.release()
cv2.destroyAllWindows()
