import argparse
import os

import cv2
import pyrootutils
from torchvision import transforms

from src.commons.model import Model

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


class RealTimePredictor:
    def __init__(self, ckpt_path):
        self.model = Model.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.model.freeze()
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True)])

    def predict(self, img):
        img = self.transforms(img)
        img = img.reshape(1, 3, 224, 224).cuda()

        prediction = self.model.forward(img)
        prediction = prediction * 80
        prediction = prediction.clip(1, 80)

        return prediction.item()


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--checkpoint_dir", help="Directory of checkpoint to load")
    args = arg_parser.parse_args()
    ckpt_path = args.checkpoint_dir
    return ckpt_path


def put_bounding_box_with_prediction(video_frame, predictor, face_classifier):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for x, y, w, h in faces:
        frame = cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        face = video_frame[y: y + h, x: x + w]
        prediction = predictor.predict(face)
        cv2.putText(frame, str(round(prediction, ndigits=2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2, )


def main():
    (ckpt_path) = parse_args()
    predictor = RealTimePredictor(ckpt_path=ckpt_path)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break

        put_bounding_box_with_prediction(video_frame, predictor, face_classifier)
        cv2.imshow("Face recognition", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
