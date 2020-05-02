import cv2
import numpy as np
import Model

face_cascade = cv2.CascadeClassifier(
    'D:\ProgramFiles\Python\lib\site-packages'
    '\cv2\data\haarcascade_frontalface_default.xml')
model = Model.load()
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame_copy = frame.copy()
    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_copy, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame_copy[y:y + h, x:x + w]
        cv2.imshow('face', face)
        face = cv2.resize(face, (32, 32))
        face = face.reshape(-1, 32, 32, 1)
        pred = model.predict_on_batch(face)
        predicted_ages = np.argmax(pred[1]) * 5
        g = np.argmax(pred[0])
        if g == 0:
            gender = 'F'
        else:
            gender = 'M'
        cv2.putText(frame,
                    str(int(predicted_ages)) + '-' + str(int(predicted_ages + 4)) + 'Y, ' + gender,
                    (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
