import cv2
import os
import numpy as np
import time

# Inisialisasi kamera
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # lebar frame
camera.set(4, 480)  # tinggi frame

# Inisialisasi classifier wajah
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi recognizer LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_label():
    data_id = {}
    for root, dirs, files in os.walk('dataset'):
        for file in files:
            if file.startswith('FaceImg.'):
                ids = int(file.split('.')[1])
                if ids not in data_id:
                    data_id[ids] = file.split('.')[2]
    return data_id

def count_registered_ids():
    registered_ids = set()
    for root, dirs, files in os.walk('dataset'):
        for file in files:
            if file.startswith('FaceImg.'):
                id_str = file.split('.')[1]
                registered_ids.add(id_str)
    return len(registered_ids)


# Fungsi untuk mengambil dataset wajah
def addFace(reg_id):

    id_name = input("Masukkan nama untuk ID wajah baru: ")
    face_id = str(reg_id + 1)

    sample_count = 0
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        frame = cv2.flip(frame,1)

        for (x, y, w, h) in faces:
            x = frame.shape[1] - (x + w) if cv2.flip(frame, 1).shape[1] > 0 else x
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample_count += 1
            # Simpan gambar wajah dalam format grayscale
            cv2.imwrite("dataset/FaceImg."+ face_id + '.' + id_name + '.' + str(sample_count) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('Adding Face Dataset', frame)
            time.sleep(1)
        
        if cv2.waitKey(1)==ord('q'):
            break
        elif sample_count >= 30:  # Ambil 30 sampel
            break
    cv2.destroyWindow('Adding Face Dataset')

# Fungsi untuk melatih recognizer LBPH
def trainRecognizer():
    faces = []
    ids = []
    dataset_path = 'dataset'
    
    # Looping semua gambar dalam dataset dan mengambil label serta wajah
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                face_img = cv2.imread(path, 0)  # Baca gambar dalam format grayscale
                face_id = int(os.path.split(path)[-1].split(".")[1])
                faces.append(face_img)
                ids.append(face_id)
    
    # Lakukan training menggunakan data wajah
    recognizer.train(faces, np.array(ids))
    recognizer.save('trained_model.yml')

# Fungsi untuk melakukan deteksi dan menampilkan kotak di sekitar wajah
def faceDetection():
    recognizer.read('trained_model.yml')
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        frame = cv2.flip(frame,1)
        
        for (x, y, w, h) in faces:
            # Lakukan prediksi menggunakan recognizer
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            x = frame.shape[1] - (x + w) if cv2.flip(frame, 1).shape[1] > 0 else x
            if confidence < 70:
                label = get_label()[id_]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1)== ord('q'):
            break


# Main program
if __name__ == "__main__":
    print('''1. Daftarkan Wajah Baru
2. Deteksi Wajah
3. Exit''')
    pilihan = int(input('Masukkan Pilihan:'))
    if pilihan == 1:
        addFace(count_registered_ids())
        trainRecognizer()
        faceDetection()
    elif pilihan ==2:
        faceDetection()
    else:
        exit()

# Tutup semua window dan kamera
camera.release()
cv2.destroyAllWindows()
