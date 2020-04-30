import numpy as np
import pickle
import cv2

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from src.face.detector import FaceDetector
from settings import LIVE_LE_PATH, LIVE_MODEL_PATH


class LiveScore:

    def __init__(self):

        self.face_detector = FaceDetector()
        self.live_model = load_model(LIVE_MODEL_PATH)
        self.live_le = pickle.loads(open(LIVE_LE_PATH, 'rb').read())

    def main(self):

        cap = cv2.VideoCapture(0)

        # loop over the frames from the video stream
        while True:
            _, frame = cap.read()

            faces = self.face_detector.detect_face(frame=frame)

            # loop over the detections
            for face_rect in faces:

                left, top, right, bottom = face_rect

                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
                face = frame[top:bottom, left:right]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = self.live_model.predict(face)[0]
                j = np.argmax(preds)
                label = self.live_le.classes_[j]

                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # show the output frame and wait for a key press
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':

    LiveScore().main()
