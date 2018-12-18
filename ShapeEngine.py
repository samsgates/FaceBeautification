import cv2
from sklearn.svm import SVC
from numpy import matrix as mat
from sklearn.decomposition import pca
from scipy import optimize
import numpy as np
import pickle
import utils

LANDMARK_NUM = 68
PCA_DIMENSIONS = 35
USE_PCA = True


class ShapeEngine:

    LANDMARK_OUTLINE = 1
    LANDMARK_LEFT_EYEBROW = 2
    LANDMARK_RIGHT_EYEBROW = 3
    LANDMARK_NOSE = 4
    LANDMARK_LEFT_EYE = 5
    LANDMARK_RIGHT_EYE = 6
    LANDMARK_MOUTH = 7

    def __init__(self):
        self.matrix = []
        self.edges = []
        self.triangles = []
        self.bound_triangles = []
        self.svm_clf = None
        self.pca_model = None
        self.knn_data = None

    @staticmethod
    def read_image(filename):
        img = cv2.imread(filename)
        rectangles = utils.detect_face(img)
        faces = []
        for rect in rectangles:
            landmarks = utils.align_face(img, rect)
            faces.append(((rect.left(), rect.top(), rect.right(), rect.bottom()), landmarks))
        return img, faces

    @staticmethod
    def get_bound_point(img, idx):
        rect = (0, 0, img.shape[1], img.shape[0])
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        idx = -idx - 1
        if idx == 0:
            return rect[0], rect[1]
        if idx == 1:
            return rect[0] + int(w / 2), rect[1]
        if idx == 2:
            return rect[0] + w - 1, rect[1]
        if idx == 3:
            return rect[0] + w - 1, rect[1] + int(h / 2)
        if idx == 4:
            return rect[0] + w - 1, rect[1] + h - 1
        if idx == 5:
            return rect[0] + int(w / 2), rect[1] + h - 1
        if idx == 6:
            return rect[0], rect[1] + h - 1
        if idx == 7:
            return rect[0], rect[1] + int(h / 2)

    @staticmethod
    def construct_triangulation(img, points):
        rect = (0, 0, img.shape[1], img.shape[0])
        sub_div = cv2.Subdiv2D(rect)
        bound_points = [ShapeEngine.get_bound_point(img, -(i + 1)) for i in range(8)]
        for point in bound_points:
            sub_div.insert(point)
        for p in points:
            sub_div.insert(p)

        triangle_list = sub_div.getTriangleList()
        triangles = []
        bound_triangles = []
        for t in triangle_list:
            pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            ind = []
            for j in range(3):
                idx = None
                for k in range(len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1e-6 and abs(pt[j][1] - points[k][1]) < 1e-6:
                        idx = k
                        break
                if idx is None:
                    for k in range(len(bound_points)):
                        if abs(pt[j][0] - bound_points[k][0]) < 1e-6 and abs(pt[j][1] - bound_points[k][1]) < 1e-6:
                            idx = -(k + 1)
                            break
                if idx is None:
                    continue
                ind.append(idx)
            if len(ind) == 3:
                if all(map(lambda x: x >= 0, ind)):
                    triangles.append(ind)
                else:
                    bound_triangles.append(ind)
        return triangles, bound_triangles

    def get_landmark_id(self, idx):
        if 0 <= idx <= 16:
            return self.LANDMARK_OUTLINE
        if 17 <= idx <= 21:
            return self.LANDMARK_LEFT_EYEBROW
        if 22 <= idx <= 26:
            return self.LANDMARK_RIGHT_EYEBROW
        if 27 <= idx <= 35:
            return self.LANDMARK_NOSE
        if 36 <= idx <= 41:
            return self.LANDMARK_LEFT_EYE
        if 42 <= idx <= 47:
            return self.LANDMARK_RIGHT_EYE
        if 48 <= idx <= 67:
            return self.LANDMARK_MOUTH

    def save_face_models(self, face_model_filename, standard_pic):
        img, faces = self.read_image(standard_pic)
        rect, landmarks = faces[0]
        triangles, bound_triangles = self.construct_triangulation(img, landmarks)
        matrix = [[0 for _ in range(LANDMARK_NUM)] for _ in range(LANDMARK_NUM)]
        with open(face_model_filename, 'wb') as model:
            for a, b, c in triangles:
                matrix[a][b] = matrix[b][a] = 10 if self.get_landmark_id(a) == self.get_landmark_id(b) else 1
                matrix[a][c] = matrix[c][a] = 10 if self.get_landmark_id(a) == self.get_landmark_id(c) else 1
                matrix[c][b] = matrix[b][c] = 10 if self.get_landmark_id(c) == self.get_landmark_id(b) else 1
            pickle.dump({
                'triangles': triangles,
                'bound_triangles': bound_triangles,
                'matrix': matrix
            }, model)

    def load_face_models(self, face_model_filename):
        with open(face_model_filename, 'rb') as model:
            model_save = pickle.load(model)
            self.matrix = model_save['matrix']
            self.triangles = model_save['triangles']
            self.bound_triangles = model_save['bound_triangles']
        assert len(self.matrix) == LANDMARK_NUM
        self.edges = []
        for i in range(LANDMARK_NUM):
            for j in range(0, i):
                if self.matrix[i][j]:
                    self.edges.append((i, j))

    def get_area(self, landmarks):
        area = 0
        for a, b, c in self.triangles:
            x1, y1 = landmarks[a]
            x2, y2 = landmarks[b]
            x3, y3 = landmarks[c]
            area += 0.5 * (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
        return area

    def get_distance_vector(self, landmarks):
        distance_vector = []
        for i, j in self.edges:
            distance = ((landmarks[i][0] - landmarks[j][0]) ** 2 + (landmarks[i][1] - landmarks[j][1]) ** 2) ** 0.5
            distance_vector.append(distance)
        sqrt_area = self.get_area(landmarks) ** 0.5
        return [distance / sqrt_area for distance in distance_vector]

    def pca_reduce(self, dv):
        return self.pca_model.transform([dv])[0] if USE_PCA else dv

    def pca_recover(self, pca_dv):
        return self.pca_model.inverse_transform(pca_dv) if USE_PCA else pca_dv

    def train_and_save_svm_model(self, paths, _labels, model_filename):
        svm_clf = SVC(C=1e3)
        vectors = []
        labels = []
        for i, (path, label) in enumerate(zip(paths, _labels)):
            if i % 10 == 1:
                print('SVM Train Parsing:', i)
            _, faces = self.read_image(path)
            if len(faces) == 0:
                print('Face Align Failed:', path)
                continue
            _, landmarks = faces[0]
            vectors.append(self.get_distance_vector(landmarks))
            labels.append(label)
        pca_model = None
        if USE_PCA:
            pca_model = pca.PCA(n_components=PCA_DIMENSIONS)
            pca_model.fit(vectors)
            vectors = pca_model.transform(vectors)
        svm_clf.fit(vectors, labels)
        with open(model_filename, 'wb') as model:
            pickle.dump({"svm_clf": svm_clf, "pca_model": pca_model}, model)

    def load_svm_model(self, model_filename):
        with open(model_filename, 'rb') as model:
            model_save = pickle.load(model)
            self.svm_clf = model_save['svm_clf']
            self.pca_model = model_save['pca_model']

    def test_svm_model(self, paths, labels):
        correct = 0
        total = 0
        for i, (path, label) in enumerate(zip(paths, labels)):
            if i % 10 == 1:
                print('SVM Train Parsing:', i)
            _, faces = self.read_image(path)
            if len(faces) == 0:
                print('Face Align Failed:', path)
                continue
            _, landmarks = faces[0]
            if self.predict_score(self.pca_reduce(self.get_distance_vector(landmarks))) == label:
                correct += 1
            total += 1
        return correct / total

    def predict_score(self, distance_vector):
        return int(self.svm_clf.predict([distance_vector]))

    def svm_generate(self, dv):
        pca_dv = self.pca_reduce(dv)
        # delta = 0.2
        # bounds = []
        # for d in pca_dv:
        #     delta_d = delta * d
        #     bounds.append((d - delta_d, d + delta_d))
        # result = optimize.minimize(lambda x: -self.predict_score(x), pca_dv, bounds=bounds)
        result = optimize.minimize(lambda x: -self.predict_score(x), pca_dv, method='BFGS')
        # result = optimize.minimize(lambda x: -self.predict_score(x), pca_dv, method='Powell')
        print(result.fun)
        print(result.success)
        return list(self.pca_recover(result.x))

    def knn_save_model(self, male_paths, male_labels, female_paths, female_labels, knn_filename):
        male_data = []
        for i, (path, label) in enumerate(zip(male_paths, male_labels)):
            if i % 10 == 1:
                print('KNN Save Parsing Male:', i)
            _, faces = self.read_image(path)
            if len(faces) == 0:
                print('Male Face Align Failed:', path)
                continue
            _, landmarks = faces[0]
            male_data.append((self.get_distance_vector(landmarks), label))
        female_data = []
        for i, (path, label) in enumerate(zip(female_paths, female_labels)):
            if i % 10 == 1:
                print('KNN Save Parsing Female:', i)
            _, faces = self.read_image(path)
            if len(faces) == 0:
                print('Female Face Align Failed:', path)
                continue
            _, landmarks = faces[0]
            female_data.append((self.get_distance_vector(landmarks), label))
        with open(knn_filename, 'wb') as out:
            pickle.dump({
                'male': male_data,
                'female': female_data
            }, out)

    def knn_load_model(self, knn_filename):
        with open(knn_filename, 'rb') as fin:
            self.knn_data = pickle.load(fin)

    def knn_generate(self, dv, gender, k=5):
        assert gender in ('male', 'female')
        ws = []
        for i, (dv_, label) in enumerate(self.knn_data[gender]):
            ws.append((label / np.linalg.norm(np.subtract(dv, dv_)), i))
        ws.sort(reverse=True)
        r = np.zeros(len(dv))
        wt = 0
        for i in range(k):
            wt += ws[i][0]
            r += ws[i][0] * np.array(self.knn_data[gender][ws[i][1]][0])
        return list(r / wt)

    def _cal_mse(self, landmarks_, dv):
        mse = 0
        for idx, (i, j) in enumerate(self.edges):
            mse += self.matrix[i][j] * (self._cal_distance(i, j, landmarks_) - dv[idx]) ** 2
        return mse

    @staticmethod
    def _cal_distance(i, j, landmarks_):
        return (landmarks_[2 * i] - landmarks_[2 * j]) ** 2 + (landmarks_[2 * i + 1] - landmarks_[2 * j + 1]) ** 2

    def _cal_partial_derivative(self, landmarks_, n):
        step = 1e-6
        landmarks_[n, 0] -= step
        y1 = np.array([self._cal_distance(i, j, landmarks_) for i, j in self.edges])
        landmarks_[n, 0] += 2 * step
        y2 = np.array([self._cal_distance(i, j, landmarks_) for i, j in self.edges])
        d = (y2 - y1) / (step * 2)
        return d

    def get_landmarks_from_dv(self, dv, landmarks):
        area = self.get_area(landmarks)
        dv = [area * d * d for d in dv]
        iteration = 0
        landmarks_ = np.array(landmarks, np.float64).reshape([2 * LANDMARK_NUM, 1])
        lase_mse = 0
        jacobi = mat(np.zeros((len(self.edges), 2 * LANDMARK_NUM)))
        u, v = 1, 2
        while iteration < 1000:
            iteration += 1
            fx = np.array([self._cal_distance(i, j, landmarks_) - dv[idx] for idx, (i, j) in enumerate(self.edges)])
            mse = self._cal_mse(landmarks_, dv)
            for j in range(2 * LANDMARK_NUM):
                jacobi[:, j] = self._cal_partial_derivative(landmarks_, j)
            h = jacobi.T * jacobi + u * np.eye(2 * LANDMARK_NUM)
            dx = -h.I * jacobi.T * fx
            landmarks_tmp = landmarks_.copy()
            landmarks_tmp += dx
            mse_tmp = self._cal_mse(landmarks_tmp, dv)
            q = float((mse - mse_tmp) / ((0.5 * dx.T * (u * dx - jacobi.T * fx))[0, 0]))
            if q > 0:
                s = 1.0 / 3.0
                v = 2
                mse = mse_tmp
                landmarks_ = landmarks_tmp
                temp = 1 - pow(2 * q - 1, 3)
                if s > temp:
                    u = u * s
                else:
                    u = u * temp
            else:
                u = u * v
                v = 2 * v
                landmarks_ = landmarks_tmp
            # print("iteration = %d, abs(mse - lase_mse) = abs(%.8f - %.8f) = %.8f"
            #       % (iteration, mse, lase_mse, abs(mse - lase_mse)))
            if abs(mse - lase_mse) < 0.000001:
                break
            lase_mse = mse
        return [tuple(map(int, map(round, landmark))) for landmark in landmarks_.reshape(LANDMARK_NUM, 2)]

    @staticmethod
    def apply_affine_transform(src, src_tri, dst_tri, size):
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        return dst

    def face_morphing(self, img, landmarks, landmarks_):
        img_morph = img.copy()
        for triangle in self.triangles + self.bound_triangles:
            t1 = list(map(lambda x: landmarks[x] if x >= 0 else self.get_bound_point(img, x), triangle))
            t2 = list(map(lambda x: landmarks_[x] if x >= 0 else self.get_bound_point(img, x), triangle))
            r1 = cv2.boundingRect(np.float32([t1]))
            r2 = cv2.boundingRect(np.float32([t2]))
            t1_rect = []
            t2_rect = []
            for j in range(0, 3):
                t1_rect.append(((t1[j][0] - r1[0]), (t1[j][1] - r1[1])))
                t2_rect.append(((t2[j][0] - r2[0]), (t2[j][1] - r2[1])))
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)
            img1rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            size = (r2[2], r2[3])
            img_rect = self.apply_affine_transform(img1rect, t1_rect, t2_rect, size)
            img_morph[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = \
                img_morph[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask) + img_rect * mask
        return img_morph

    def some_landmarks_only(self, landmarks, landmarks_, id_set):
        r = landmarks.copy()
        for i in range(LANDMARK_NUM):
            if self.get_landmark_id(i) in id_set:
                r[i] = landmarks_[i]
        return r

    def some_landmarks_out(self, landmarks, landmarks_, id_set):
        r = landmarks.copy()
        for i in range(LANDMARK_NUM):
            if self.get_landmark_id(i) not in id_set:
                r[i] = landmarks_[i]
        return r

    def eyes_only(self, landmarks, landmarks_):
        return self.some_landmarks_only(landmarks, landmarks_, (self.LANDMARK_LEFT_EYE, self.LANDMARK_RIGHT_EYE))

    def outline_only(self, landmarks, landmarks_):
        return self.some_landmarks_only(
            landmarks, landmarks_,
            (self.LANDMARK_OUTLINE, self.LANDMARK_LEFT_EYEBROW, self.LANDMARK_RIGHT_EYEBROW))

    def make_bigger_eyes(self, landmarks, rate):
        distance_vector = []
        for i, j in self.edges:
            distance = ((landmarks[i][0] - landmarks[j][0]) ** 2 + (landmarks[i][1] - landmarks[j][1]) ** 2) ** 0.5

            f = (self.get_landmark_id(i) == self.LANDMARK_LEFT_EYE) \
                + (self.get_landmark_id(j) == self.LANDMARK_LEFT_EYE)
            if f == 0:
                f = (self.get_landmark_id(i) == self.LANDMARK_RIGHT_EYE) \
                    + (self.get_landmark_id(j) == self.LANDMARK_RIGHT_EYE)
            if f == 1:
                distance = distance * (1 - rate)
            if f == 2:
                distance = distance * (1 + rate)
            distance_vector.append(distance)
        sqrt_area = self.get_area(landmarks) ** 0.5
        dv = [distance / sqrt_area for distance in distance_vector]
        return self.get_landmarks_from_dv(dv, landmarks)

    def make_thinner_outline(self, landmarks, rate):
        def check(x):
            return 0 <= x <= 16
        distance_vector = []
        for i, j in self.edges:
            distance = ((landmarks[i][0] - landmarks[j][0]) ** 2 + (landmarks[i][1] - landmarks[j][1]) ** 2) ** 0.5

            # f = (self.get_landmark_id(i) == self.LANDMARK_OUTLINE) \
            #     + (self.get_landmark_id(j) == self.LANDMARK_OUTLINE)

            f = check(i) + check(j)
            if f == 1:
                distance = distance * (1 - rate)
            distance_vector.append(distance)
        sqrt_area = self.get_area(landmarks) ** 0.5
        dv = [distance / sqrt_area for distance in distance_vector]
        return self.get_landmarks_from_dv(dv, landmarks)
