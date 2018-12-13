import cv2
from ShapeEngine import ShapeEngine
import utils

engine = ShapeEngine()

CONNECTIVITY_MATRIX_FILE = 'model/matrix.txt'
TRIANGLE_LIST_FILE = 'model/triangle.txt'
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
    edges = engine.get_connect_edges()

    img, faces = engine.read_image(path)
    rect, landmarks = faces[0]

    for i, j in edges:
        cv2.line(img, landmarks[i], landmarks[j],
                 (255, 0, 0) if engine.get_landmark_id(i) == engine.get_landmark_id(j) else (0, 0, 255),
                 1, cv2.LINE_AA)

    print(len(engine.get_distance_vector(landmarks)))
    cv2.imshow('img', img)
    cv2.waitKey()


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
        paths, labels = load_data_file('Data/train_images.txt', 'Data/Images/%s')
        male_paths = []
        male_labels = []
        for path, label in zip(paths, labels):
            if path[13] == 'M':
                male_paths.append(path)
                male_labels.append(label)
        print('Path num:', len(male_paths))
        engine.knn_save_model(paths, labels, KNN_MODEL_FILE)
    engine.knn_load_model(KNN_MODEL_FILE)


def knn_beautify(filename):
    img, faces = engine.read_image(filename)
    rect, landmarks = faces[0]
    dv_ = engine.knn_generate(engine.get_distance_vector(landmarks))
    landmarks_ = engine.get_landmarks_from_dv(dv_, landmarks)
    img_morph = engine.face_morphing(img, landmarks, landmarks_)
    return img, img_morph


def svm_beautify(filename):
    img, faces = engine.read_image(filename)
    rect, landmarks = faces[0]
    dv_ = engine.svm_generate(engine.get_distance_vector(landmarks))
    landmarks_ = engine.get_landmarks_from_dv(dv_, landmarks)
    img_morph = engine.face_morphing(img, landmarks, landmarks_)
    return img, img_morph


def main():
    # engine.save_face_models(CONNECTIVITY_MATRIX_FILE, TRIANGLE_LIST_FILE, 'Data/img/001.jpg')
    engine.load_face_models(CONNECTIVITY_MATRIX_FILE, TRIANGLE_LIST_FILE)

    # show_image('Data/img/002.jpg')

    # init_svm(train=True, test=True)
    init_svm(train=False, test=False)
    img, img_morph = svm_beautify('Data/img/134.jpg')

    # init_knn(train=False)
    # img, img_morph = knn_beautify('Data/img/005.jpg')

    utils.plt_show(img)
    utils.plt_show(img_morph)


if __name__ == '__main__':
    main()
