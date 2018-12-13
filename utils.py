import dlib
import cv2
from matplotlib import pyplot as plt

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')


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
    for landmark in landmarks:
        cv2.circle(img, landmark, 2, bgr_color, cv2.FILLED, cv2.LINE_AA, 0)


def main():
    img = cv2.imread("Test/test.jpg")
    rectangles = detect_face(img)
    for rect in rectangles:
        landmarks = align_face(img, rect)
        print(rect.left(), rect.top(), rect.right(), rect.bottom())
        draw_landmark(img, landmarks, (0, 0, 255))
    plt_show(img)


if __name__ == '__main__':
    main()
