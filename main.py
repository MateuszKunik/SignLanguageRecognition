import numpy as np
import cv2
from mediapipe import solutions
from collections import OrderedDict
from tensorflow.keras.models import load_model


LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
FULL_PATH = "./AmericanSignLanguage/Models/best_model.h5"
model = load_model(FULL_PATH)

TEXT_COLOR = (90, 140, 0)
TEXT_THICKNESS = 2

FRAME_COLOR = (90, 140, 50)
FRAME_CONTOUR = (90, 140, 50)
FRAME_THICKNESS = 6



def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def main_distance(x, y):
    x0 = x[0]
    y0 = y[0]
    
    x1 = np.mean([value for index, value in enumerate(x) if index % 4 == 1 and index != 1])
    y1 = np.mean([value for index, value in enumerate(y) if index % 4 == 1 and index != 1])
    
    distance = euclidean_distance((x0, y0), (x1, y1))
    
    return distance


def subtract(point, margin):
    point = np.subtract(point, margin)
    
    for pos, coordinate in enumerate(point):
        if coordinate < 0:
            point[pos] = 0
    
    return point


def add(point, margin, image_shape):
    point = np.add(point, margin)
    image_shape = np.flip(image_shape)
    
    for pos, coordinate in enumerate(point):
        if coordinate > image_shape[pos]:
            point[pos] = image_shape[pos]
    
    return point


def get_bbox_coordinates(handLandmark, image_shape):
    x, y = [], []
    
    for hand in mp_hands.HandLandmark:
        x.append(int(handLandmark.landmark[hand].x * image_shape[1]))
        y.append(int(handLandmark.landmark[hand].y * image_shape[0]))
    
    start_point = (min(x), min(y))
    end_point = (max(x), max(y))
     
    return x, y, start_point, end_point


def visual_bbox(image_shape, start_point, end_point, measure):
    margin = 2 * round(np.sqrt(measure))
    
    start_point = subtract(start_point, margin)
    end_point = add(end_point, margin, image_shape)
    
    return tuple(start_point), tuple(end_point)


def convert2square(region):
    shape = region.shape[:2]
    
    if shape[0] != shape[1]:
        main_size = max(shape) 
        index = shape.index(main_size)

        difference = main_size - min(shape)
        thickness = difference//2

        if index == 0:
            x = thickness
            y = 0
        else:
            x = 0
            y = thickness

        return cv2.copyMakeBorder(region, y, y, x, x, borderType = cv2.BORDER_REPLICATE)
    
    else:
        return region


def prediction(region, model, class_names):
    region = cv2.resize(region, (28, 28))
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    x = region.reshape(1, 28, 28, 1)
    pred = model.predict(x)
    indicator = pred.argmax()
    
    sign = np.array(class_names)[indicator]
    proba = pred[np.arange(pred.shape[0]), indicator][0]
    
    return sign, proba




mp_hands = solutions.hands
cap = cv2.VideoCapture(0)
cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)

with mp_hands.Hands(
    min_detection_confidence = 0.9,
    min_tracking_confidence = 0.5,
    max_num_hands = 1) as hands:
    while cap.isOpened():
        _, frame = cap.read()
        image_shape = frame.shape[:2]
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # getting info about landmarks and creating main bounding box
                x, y, start_point, end_point = get_bbox_coordinates(hand, image_shape)
                #cv2.rectangle(image, start_point, end_point, frame_color, 3)
                
                # creating visual bounding box
                distance = main_distance(x, y)
                frame_thickness = int(distance//20)
                vstart_point, vend_point = visual_bbox(image_shape, start_point, end_point, distance)
                cv2.rectangle(image, vstart_point, vend_point, FRAME_COLOR, FRAME_THICKNESS)
                
                # creating region of interest for predictions
                roi = frame[vstart_point[1]:vend_point[1], vstart_point[0]:vend_point[0]]
                roi = convert2square(roi)
                
                # making predictions
                sign, probability = prediction(roi, model, LETTERS)             
                cv2.putText(image, f"{sign}" + " %.2f" % probability, vstart_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, TEXT_THICKNESS, FRAME_COLOR, TEXT_THICKNESS)

        cv2.imshow("Sign Language Recognition", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    
    
cap.release()
cv2.destroyAllWindows()