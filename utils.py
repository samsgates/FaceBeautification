import dlib
import cv2
from matplotlib import pyplot as plt
import numpy as np

GENDER_FEMALE = 0
GENDER_MALE = 1
_USE_GENDER = False

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
if _USE_GENDER:
    from keras.models import load_model
    gender_classifier = load_model("model/gender_mini_XCEPTION.21-0.95.hdf5", compile=False)
else:
    gender_classifier = None


def gender_predict(img, rect):
    x, y, w, h = rect
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = img_gray[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (64, 64))
    face = np.expand_dims(face, 0)
    face = np.expand_dims(face, -1)
    face = face / 255.0
    return int(np.argmax(gender_classifier.predict(face)))


def detect_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rectangles = _detector(img_gray, 0)
    return rectangles


def align_face(img, rect):
    landmarks = [(p.x, p.y) for p in _predictor(img.copy(), rect).parts()]
    return landmarks


def plt_show(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def draw_landmark(img, landmarks, bgr_color):
    for i, landmark in enumerate(landmarks):
        # cv2.putText(img, str(i + 1), landmark, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        cv2.circle(img, landmark, 2, bgr_color, cv2.FILLED, cv2.LINE_AA, 0)


def main():
    img = cv2.imread("Test/test.jpg")
    rectangles = detect_face(img)
    for rect in rectangles:
        landmarks = align_face(img, rect)
        print(rect.left(), rect.top(), rect.right(), rect.bottom(), rect.width(), rect.height())
        draw_landmark(img, landmarks, (0, 0, 255))
        print('female' if gender_predict(img, (rect.left(), rect.top(), rect.width(), rect.height())) == GENDER_FEMALE
              else 'male')
    plt_show(img)


if __name__ == '__main__':
    main()
