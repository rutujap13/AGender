from keras.utils import np_utils
from scipy.io import loadmat
from datetime import datetime
import os
import numpy as np
import cv2
import scipy.io
from tqdm import tqdm


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass

    
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def find_face(gray):
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (32, 32))
        return True, face
    return False, gray


def process_dataset():
    output_path = 'Dataset/dataset.mat'
    db = 'imdb'
    img_size = 32
    ch=1
    min_score = 1.0
    root_path = "{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    out_genders = []
    out_ages = []
    sample_num = len(face_score)
    out_imgs = np.empty((sample_num, img_size, img_size, ch), dtype=np.uint8)
    valid_sample_num = 0
    for i in tqdm(range(sample_num)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <70):
            continue

        if np.isnan(gender[i]):
            continue
        img = cv2.imread(root_path + str(full_path[i][0]),cv2.IMREAD_GRAYSCALE)
        flag,img = find_face(img)
        if not flag:
            continue
        out_imgs[valid_sample_num] = img.reshape(1,32,32,1)
        out_genders.append(int(gender[i]))
        out_ages.append(int(age[i]/5))
        valid_sample_num += 1

    output = {"image": out_imgs[:valid_sample_num], "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    scipy.io.savemat(output_path, output)


def load_dataset():
    if not os.path.exists('Dataset/dataset.mat'):
        process_dataset()
    image, gender, age, _, image_size, _ = load_data('Dataset/dataset.mat')
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    y_data_a = np_utils.to_categorical(age, 14)
    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - 0.1))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]
    return X_train, X_test, y_train_a, y_train_g, y_test_a, y_test_g


if __name__ == '__main__':
    load_dataset()
