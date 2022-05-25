import sys
sys.path.insert(1,'C:/Users/gopal/Downloads/source-code-face-recognition/sourcecode')
import datetime
from ctypes import *
import random
import os
import csv
import PIL.Image
import cv2
import time
import darknet
import argparse
from PIL import Image
from threading import Thread, enumerate
from queue import Queue
import os
import pytesseract
import numpy as np
import datetime
from simple_facerec import SimpleFacerec


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    '''parser.add_argument("--input", type=str, default="C:/Users/gopal/Downloads/video.MOV",
                        help="video source. If empty, uses webcam 0 stream")'''
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))

    count=i=0
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        extract_num(frame)
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                if str(label)=='person'or str(label)=='car'or str(label)=='bicycle'or str(label)=='bus'or str(label)=='truck'or str(label)=='motorbike' or str(label)=='sports ball':
                    detections_adjusted.append((str(label), confidence, bbox_adjusted))
            #print(bbox_adjusted)
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            #ret, frame1 = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Width: ' + str(cap.get(3)) + ' Height:' + str(cap.get(4))
            datet = str(datetime.datetime.now())
            """frame = cv2.putText(frame, text, (10, 100), font, 1,
                                    (0, 255, 255), 2, cv2.LINE_AA)"""
            frame = cv2.putText(frame, datet, (10, 25), font, 0.5,
                                    (0, 255, 255), 2, cv2.LINE_AA)
            # Detect Faces
            padding = 20
            face_locations, face_names, face_times = sfr.detect_known_faces(frame)
            for face_loc, name, times in zip(face_locations, face_names, face_times):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                # cv2.putText(frame, times, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                print(times)
                writer.writerow(times)
                #face=frame[y1:x1,x2:y2]
                #face = frame[max(0, face_loc[1] - padding):min(face_loc[3] + padding, frame.shape[0] - 1),
                       #max(0, face_loc[0] - padding):min(face_loc[2] + padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPred = genderNet.forward()
                gender = genderList[genderPred[0].argmax()]

                ageNet.setInput(blob)
                agePred = ageNet.forward()
                age = ageList[agePred[0].argmax()]

                label = "{},{}".format(gender, age)
                cv2.rectangle(frame, (face_loc[0], face_loc[1] - 30), (face_loc[2], face_loc[1]), (0, 255, 0), -1)
                cv2.putText(frame, label, (face_loc[0], face_loc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
            #cv2.imshow("Age-Gender", frame)
            '''count += 1
            x, y, z, h = darknet.bbox2points(bbox_adjusted)
            imcap = image[x:h, y:z]
            if(count>=0 and float(confidence)>99):
                cv2.imwrite(os.path.join("C:/Users/gopal/OneDrive/Desktop/darknet-master/pics",'Frame' + str(datetime.datetime.now()) + '.jpg'), imcap)
            i+=1
            count=0'''

            data=(str(label),str((datetime.datetime.now().strftime("%H:%M:%S"))),str(datetime.datetime.today().strftime("%Y:%m:%d")))
            print(data)
            writer.writerow(data)
            '''r = open('timeline.csv', 'r')
            reader = r.readlines(-1)
            if (data[1] - reader[1]):
                print(str(reader))'''
            cv2.imshow('Inference', image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()
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
    #nplate = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml').detectMultiScale(gray, 1.1, 4)
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
            cv2.imwrite(os.path.join("C:/Users/gopal/OneDrive/Desktop/darknet-master/pics", 'plate' + read + '.jpg'), plate)
        else:
            cv2.imwrite(os.path.join("C:/Users/gopal/OneDrive/Desktop/darknet-master/pics",'plateUNKNOWN.jpg'),plate)

    # cv2.imshow("Result", img)
    # cv2.imwrite('result.jpg',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # Encode faces from a folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    states = {"AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
              "BR": "Bihar",
              "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu", "DL": "Delhi", "GA": "Goa",
              "GJ": "Gujarat",
              "HR": "Haryana", "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
              "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya",
              "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odissa", "PY": "Pondicherry", "PN": "Punjab", "RJ": "Rajasthan",
              "SK": "Sikkim", "TN": "TamilNadu", "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal",
              "CG": "Chhattisgarh", "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"}
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    data = []
    f=open("timeline.csv",'w')
    writer=csv.writer(f)
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()