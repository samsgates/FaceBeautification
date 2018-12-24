from PyQt5.QtGui import QImage, QPixmap
import cv2
import utils
from ShapeEngine import ShapeEngine
from BeverageRemoving.Bilateral_filtering import Bilateral_filter
from Whitening.Whitening import Whitening


FACE_MODEL_FILE = 'model/face.model'
KNN_MODEL_FILE = 'model/knn.model'


class FaceBeautification:

    def __init__(self):
        self.shape_engine = ShapeEngine()
        self.shape_engine.load_face_models(FACE_MODEL_FILE)
        self.shape_engine.knn_load_model(KNN_MODEL_FILE)
        self.sequence = [-1, []]
        self.img_rect = None

    def current_sequence(self):
        assert 0 <= self.sequence[0] < len(self.sequence[1])
        return self.sequence[1][self.sequence[0]]

    def prev_sequence(self):
        assert 0 < self.sequence[0] < len(self.sequence[1])
        self.sequence[0] -= 1
        return self.sequence[1][self.sequence[0]]

    def next_sequence(self):
        assert 0 <= self.sequence[0] < len(self.sequence[1]) - 1
        self.sequence[0] += 1
        return self.sequence[1][self.sequence[0]]

    def is_sequence_empty(self):
        return self.sequence[0] == -1

    def at_sequence_front(self):
        return self.sequence[0] == 0

    def at_sequence_end(self):
        return self.sequence[0] == len(self.sequence[1]) - 1

    def clear_sequence(self):
        self.sequence = [-1, []]

    def reset_sequence(self):
        self.sequence = [0, [self.sequence[1][0]]]

    def add_to_sequence(self, img, landmarks):
        if self.sequence[0] == -1:
            self.sequence[0] = 0
            self.sequence[1].append((img, landmarks))
        else:
            self.sequence[1] = self.sequence[1][:self.sequence[0] + 1]
            self.sequence[1].append((img, landmarks))
            self.sequence[0] += 1
            assert self.sequence[0] == len(self.sequence[1]) - 1

    def load_image(self, filename):
        img = cv2.imread(filename)
        rectangles = utils.detect_face(img)
        if not rectangles:
            return False
        rect = rectangles[0]
        landmarks = utils.align_face(img, rect)
        self.img_rect = (rect.left(), rect.top(), rect.width(), rect.height())
        self.clear_sequence()
        self.add_to_sequence(img, landmarks)
        return True

    def save_image(self, filename):
        cv2.imwrite(filename, self.current_sequence()[0])

    def get_original_image(self):
        assert self.sequence[0] != -1
        return self._get_qt_pix_map(self.sequence[1][0][0])

    def get_beautified_image(self):
        return self._get_qt_pix_map(self.current_sequence()[0])

    @staticmethod
    def _get_qt_image(img):
        if img is None:
            return None
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return QImage(img_[:], img_.shape[1], img_.shape[0], img_.shape[1] * 3, QImage.Format_RGB888)

    @staticmethod
    def _get_qt_pix_map(img):
        img_ = FaceBeautification._get_qt_image(img)
        if img_ is None:
            return None
        return QPixmap().fromImage(img_)

    def apply_knn(self, gender, eyebrows, eyes, nose, mouth, outline):
        features = []
        if eyebrows:
            features.append(ShapeEngine.LANDMARK_LEFT_EYEBROW)
            features.append(ShapeEngine.LANDMARK_RIGHT_EYEBROW)
        if eyes:
            features.append(ShapeEngine.LANDMARK_LEFT_EYE)
            features.append(ShapeEngine.LANDMARK_RIGHT_EYE)
        if nose:
            features.append(ShapeEngine.LANDMARK_NOSE)
        if mouth:
            features.append(ShapeEngine.LANDMARK_MOUTH)
        if outline:
            features.append(ShapeEngine.LANDMARK_OUTLINE)
        img, landmarks = self.current_sequence()
        dv_ = self.shape_engine.knn_generate(self.shape_engine.get_distance_vector(landmarks), gender)
        landmarks_ = self.shape_engine.get_landmarks_from_dv(dv_, landmarks)
        landmarks_ = self.shape_engine.some_landmarks_only(landmarks, landmarks_, features)
        img_morph = self.shape_engine.face_morphing(img, landmarks, landmarks_)
        self.add_to_sequence(img_morph, landmarks_)

    def apply_bigger_eyes(self, rate):
        img, landmarks = self.current_sequence()
        landmarks_ = self.shape_engine.make_bigger_eyes(landmarks, rate)
        landmarks_ = self.shape_engine.eyes_only(landmarks, landmarks_)
        img_morph = self.shape_engine.face_morphing(img, landmarks, landmarks_)
        self.add_to_sequence(img_morph, landmarks_)

    def apply_thinner_eyes(self, rate):
        img, landmarks = self.current_sequence()
        landmarks_ = self.shape_engine.make_thinner_outline(landmarks, rate)
        landmarks_ = self.shape_engine.outline_only(landmarks, landmarks_)
        img_morph = self.shape_engine.face_morphing(img, landmarks, landmarks_)
        self.add_to_sequence(img_morph, landmarks_)

    def apply_remove_beverage(self):
        img, landmarks = self.current_sequence()
        img_ = Bilateral_filter(img.copy(), 0, 0, img.shape[0], img.shape[1])
        self.add_to_sequence(img_, landmarks)

    def apply_whitening(self):
        img, landmarks = self.current_sequence()
        img_ = Whitening(img.copy(), *self.img_rect)
        self.add_to_sequence(img_, landmarks)

    def apply_facelet(self):
        pass

def main():
    print(type(cv2.imread('Test/test.jpg')))


if __name__ == '__main__':
    main()
