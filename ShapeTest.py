import cv2
from ShapeEngine import ShapeEngine
import utils

engine = ShapeEngine()

FACE_MODEL_FILE = 'model/face.model'
SVM_MODEL_FILE = 'model/svm.model'
KNN_MODEL_FILE = 'model/knn.model'


def load_data_file(path, filename_format):
    with open(path) as fin:
        paths = []
        labels = []
        for line in fin:
            path, label = line.split()[:2]
            paths.append(filename_format % path)
            labels.append(int(round(float(label))))
        return paths, labels


def show_image(path):
    # edges = engine.edges

    img, faces = engine.read_image(path)
    _, landmarks = faces[0]

    # def get_color(idx1, idx2):
    #     f = (engine.get_landmark_id(idx1) == engine.LANDMARK_LEFT_EYE) \
    #         + (engine.get_landmark_id(idx2) == engine.LANDMARK_LEFT_EYE)
    #     if f == 0:
    #         f = (engine.get_landmark_id(idx1) == engine.LANDMARK_RIGHT_EYE) \
    #             + (engine.get_landmark_id(idx2) == engine.LANDMARK_RIGHT_EYE)
    #     if f == 0:
    #         return 0, 0, 255
    #     if f == 1:
    #         return 255, 0, 0
    #     if f == 2:
    #         return 0, 255, 0
    #     return 255, 255, 255
    #
    # for i, j in edges:
    #     cv2.line(img, landmarks[i], landmarks[j], get_color(i, j), 1, cv2.LINE_AA)

    for a, b, c in engine.triangles:
        cv2.line(img, landmarks[a], landmarks[b], (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, landmarks[a], landmarks[c], (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, landmarks[c], landmarks[b], (0, 0, 255), 1, cv2.LINE_AA)

    for triangle in engine.bound_triangles:
        ps = list(map(lambda x: landmarks[x] if x >= 0 else engine.get_bound_point(img, x), triangle))
        # print(ps[0], ps[1], ps[2])
        cv2.line(img, ps[0], ps[1], (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, ps[0], ps[2], (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, ps[2], ps[1], (0, 255, 0), 1, cv2.LINE_AA)

    utils.plt_show(img)


def init_svm(train=False, test=False):
    import DataManager
    DataManager.stretch_score('Data/All_labels.txt', 'Data/All_labels_stretch.txt', 1, 20)
    DataManager.partition_data('Data/All_labels_stretch.txt',
                               'Data/train_images_stretch.txt',
                               'Data/test_images_stretch.txt')
    # train_paths, train_labels = load_data_file('Data/train.txt', 'Data/img/%s.jpg')
    # test_paths, test_labels = load_data_file('Data/test.txt', 'Data/img/%s.jpg')
    if train:
        train_paths, train_labels = load_data_file('Data/train_images_stretch.txt', 'Data/Images/%s')
        print('training and saving...')
        engine.train_and_save_svm_model(train_paths, train_labels, SVM_MODEL_FILE)
    print('loading svm model...')
    engine.load_svm_model(SVM_MODEL_FILE)
    if test:
        print('testing...')
        test_paths, test_labels = load_data_file('Data/test_images_stretch.txt', 'Data/Images/%s')
        accuracy = engine.test_svm_model(test_paths, test_labels)
        print('accuracy:', accuracy)


def init_knn(train=False):
    if train:
        # paths, labels = load_data_file('Data/train.txt', 'Data/img/%s.jpg')
        # paths, labels = load_data_file('Data/train_images.txt', 'Data/Images/%s')
        male_paths, male_labels = load_data_file('Data/male_labels.txt', '%s')
        female_paths, female_labels = load_data_file('Data/female_labels.txt', '%s')
        print('Male Path num:', len(male_paths))
        print('Female Path num:', len(female_paths))
        engine.knn_save_model(male_paths, male_labels, female_paths, female_labels, KNN_MODEL_FILE)
    engine.knn_load_model(KNN_MODEL_FILE)


def knn_beautify(filename, gender):
    img, faces = engine.read_image(filename)
    _, landmarks = faces[0]
    dv_ = engine.knn_generate(engine.get_distance_vector(landmarks), gender)
    landmarks_ = engine.get_landmarks_from_dv(dv_, landmarks)
    # landmarks_ = engine.outline_only(landmarks, landmarks_)
    # landmarks_ = engine.some_landmarks_out(landmarks, landmarks_, (engine.LANDMARK_MOUTH, ))
    img_morph = engine.face_morphing(img, landmarks, landmarks_)
    return img, img_morph


def svm_beautify(filename):
    img, faces = engine.read_image(filename)
    _, landmarks = faces[0]
    dv_ = engine.svm_generate(engine.get_distance_vector(landmarks))
    landmarks_ = engine.get_landmarks_from_dv(dv_, landmarks)
    img_morph = engine.face_morphing(img, landmarks, landmarks_)
    return img, img_morph


def make_bigger_eyes(filename, rate):
    img, faces = engine.read_image(filename)
    _, landmarks = faces[0]
    landmarks_ = engine.make_bigger_eyes(landmarks, rate)
    landmarks_ = engine.eyes_only(landmarks, landmarks_)
    img_morph = engine.face_morphing(img, landmarks, landmarks_)
    return img, img_morph


def make_thinner_outline(filename, rate):
    img, faces = engine.read_image(filename)
    _, landmarks = faces[0]
    landmarks_ = engine.make_thinner_outline(landmarks, rate)
    landmarks_ = engine.outline_only(landmarks, landmarks_)
    img_morph = engine.face_morphing(img, landmarks, landmarks_)
    return img, img_morph


def main():
    # engine.save_face_models(FACE_MODEL_FILE, 'Data/img/005.jpg')
    engine.load_face_models(FACE_MODEL_FILE)

    # img_filename = 'Data/img/005.jpg'
    img_filename = 'Test/0007_01.jpg'

    # show_image(img_filename)

    # init_svm(train=True, test=True)
    # init_svm(train=False, test=False)
    # img, img_morph = svm_beautify(img_filename)

    init_knn(train=False)
    img, img_morph = knn_beautify(img_filename, 'male')

    # img, img_morph = make_bigger_eyes(img_filename, 0.05)

    # img, img_morph = make_thinner_outline(img_filename, 0.1)

    utils.plt_show(img)
    utils.plt_show(img_morph)


if __name__ == '__main__':
    main()
