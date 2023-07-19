# Import the libraries
from ultralytics import YOLO
import cv2
import cvzone
import random
# import math
import json
import os
# import mysql.connector
from datetime import datetime
# import matplotlib.pyplot as plt
import pytesseract
from preprocessmodule import ImagePreprocessor
from sqlalchemy import create_engine, Column, Integer, String, BLOB, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get the config file
with open('../configfiles/config.json') as config_file:
    config = json.load(config_file)

# Make a directory for saved images
try:
    os.mkdir(config["paths"]["saved_Images_path"])
except OSError as error:
    print(error)


# Create an ORM for our Database
connection_string = f'mysql+pymysql://{config["mysql"]["user"]}:{config["mysql"]["password"]}@' \
                    f'{config["mysql"]["host"]}/{config["mysql"]["database"]}'
engine = create_engine(connection_string, echo=True)
Base = declarative_base()


class Data(Base):
    __tablename__ = "Data"
    Data_Id = Column(Integer, primary_key=True, autoincrement=True)
    Predicted_Frame = Column(BLOB, nullable=False)
    OCR_Text = Column(String(50), nullable=False)
    Histogram = Column(BLOB, nullable=False)
    OCR_Frame = Column(BLOB, nullable=False)
    Date_of_Upload = Column(Date, nullable=False)
    Accuracy = Column(Float, nullable=False)
    Model_Name = Column(String(50), nullable=False)

    def __init__(self, Predicted_Frame, OCR_Text, Histogram, OCR_Frame, Date_of_Upload, Accuracy, Model_Name):
        self.Predicted_Frame = Predicted_Frame
        self.OCR_Text = OCR_Text
        self.Histogram = Histogram
        self.OCR_Frame = OCR_Frame
        self.Date_of_Upload = Date_of_Upload
        self.Accuracy = Accuracy
        self.Model_Name = Model_Name

    def __repr__(self):
        return f"({self.Data_Id}) {self.OCR_Text} {self.Accuracy} {self.Model_Name}"


# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Create an object for pre-process operation
filter = ImagePreprocessor()


# Define a pre-process function for frames to get better results from OCR
def preprocess(frm):
    rescaled = filter.rescale(frm, 0.75)
    gray = filter.gray_filter(rescaled)
    thresholded = filter.thresholding(gray, threshold_value=170)
    thin_font = filter.thin_font(thresholded)
    no_borders = filter.remove_borders(thin_font)

    return no_borders


# Open the camera
cap = cv2.VideoCapture(config["videoCapture"]["device"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["videoCapture"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["videoCapture"]["height"])

# Define the YOLO Model
model = YOLO(config["paths"]["model_Path"])

# Define detection and database classes
classNames = config["classes"]["classNames"]
sqlClasses = config["classes"]["sqlClasses"]

# Get frames from camera
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)  # Apply the model to frames
        for_text = frame.copy()
        for r in results:  # Detect the objects
            boxes = r.boxes
            for box in boxes:
                cls = classNames[int(box.cls[0])]
                if cls in sqlClasses:
                    try:  # Check if detected object in our database or not
                        result = session.query(Data.Accuracy, Data.OCR_Text).filter(Data.Model_Name == cls).all()
                        if result:
                            for rs in result:
                                print(
                                    f"OCR Text: {rs[1]}\n"
                                    f"Database Oranı: {rs[0]}\n"
                                    f"Kamera Oranı: {round(float(box.conf[0]), 3)}")
                        else:
                            print(f"{classNames[int(box.cls[0])].upper()} Verisi Database'de bulunamadı.")
                    except Exception as error:
                        print(error)

                # Get detected object's bounding boxes
                x1, y1, x2, y2 = box.xyxy[0]
                w, h = int(x2) - int(x1), int(y2) - int(y1)
                bbox = int(x1), int(y1), w, h

                # Draw a rectangle to detected objects
                cvzone.cornerRect(frame, bbox, l=config["rectSetup"]["length"], t=config["rectSetup"]["thickness"],
                                  colorR=tuple(config["rectSetup"]["rectColor"]))

                # Get detected object's confidence score
                conff = round(float(box.conf[0]), 3)

                # Get the type of the detected object
                cls = box.cls[0]
                crClass = classNames[int(cls)]
                cvzone.putTextRect(frame, f'{crClass.upper()} {conff}', (max(0, int(x1)), max(35, int(y1))),
                                   scale=config["textSetup"]["scale"], thickness=config["textSetup"]["thickness"],
                                   offset=config["textSetup"]["offset"])

                # Add a button
                key = cv2.waitKey(1)
                if key == ord("s"):  # If press "s", take a screenshot and save the frame
                    cv2.imwrite(config["paths"]["saved_Images_path"] + "/saved_Image_{}".format(str(random.random)),
                                frame)

                elif key == ord("w"):  # If press "w", insert the following data into our database
                    title = config["paths"]["saved_Images_path"] + "/saved_Image_{}.jpg".format(str(random.random()))
                    cv2.imwrite(title, frame)

                    with open(title, "rb") as frame_file:
                        binary_frame = frame_file.read()

                    time = datetime.now()
                    date = f"{time.year}-{time.month:02d}-{time.day:02d}"

                    title_text = config["paths"]["saved_Images_path"] + "/saved_Image_{}.jpg".format(
                        str(random.random()))
                    cv2.imwrite(title_text, for_text)
                    with open(title_text, "rb") as frame_text:
                        binary_text = frame_text.read()

                    OCR_text = pytesseract.image_to_string(for_text)

                    histg = cv2.calcHist([frame], [0], None, [256], [0, 256])
                    histg_bytes = histg.tobytes()

                    new_data = Data(binary_frame, OCR_text, histg_bytes, binary_text, date, conff, crClass)

                    session.add(new_data)
                    session.commit()

                    os.remove(title)
                    os.remove(title_text)

                elif key == ord("t"):  # If press "t", show the detected text
                    text = pytesseract.image_to_string(frame)
                    print(text)

        # Get stream with camera
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # If press "q", close the camera
            break

else:
    print("Kamera Açılamıyor!")

cap.release()
cv2.destroyAllWindows()
