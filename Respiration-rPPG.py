import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk membuat low-pass filter sederhana menggunakan moving average
def moving_average_filter(data, window_size=5):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

# Variabel untuk sinyal respirasi dan rPPG
respiration_signal = []
rPPG_signal = []
rPPG_signal_filtered = []

# Parameter untuk pemrosesan sinyal
respiration_freq = 0.2  # frekuensi pernapasan dalam Hz
rPPG_freq = 1.5  # frekuensi detak jantung dalam Hz
window_size = 5  # ukuran window untuk filter moving average

# Membuat plot untuk visualisasi
plt.ion()  # interactive mode untuk pembaruan grafik
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
x_axis = np.arange(0, 100, 1)

# Fungsi untuk menghitung sinyal rPPG
def calculate_rPPG(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)
    return avg_brightness

# Fungsi untuk menghitung sinyal respirasi
def calculate_respiration_signal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = frame[y:y+h, x:x+w]
        avg_color = np.mean(roi, axis=(0, 1))
        return avg_color[1]
    return 0

# Proses video secara real-time
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Menghitung sinyal rPPG dan respirasi
    rPPG_val = calculate_rPPG(frame)
    respiration_val = calculate_respiration_signal(frame)

    # Tambahkan nilai sinyal ke daftar
    rPPG_signal.append(rPPG_val)
    respiration_signal.append(respiration_val)
    
    # Filter sinyal rPPG menggunakan moving average
    if len(rPPG_signal) > window_size:
        rPPG_signal_filtered = moving_average_filter(np.array(rPPG_signal), window_size)
    else:
        rPPG_signal_filtered = rPPG_signal
    
    # Visualisasikan sinyal
    ax1.clear()
    ax1.plot(x_axis[:len(rPPG_signal_filtered)], rPPG_signal_filtered, label="rPPG Signal")
    ax1.set_title("Sinyal rPPG")
    ax1.set_xlabel("Waktu (detik)")
    ax1.set_ylabel("Amplitudo")
    ax1.legend()

    ax2.clear()
    ax2.plot(x_axis[:len(respiration_signal)], respiration_signal, label="Sinyal Respirasi")
    ax2.set_title("Sinyal Respirasi")
    ax2.set_xlabel("Waktu (detik)")
    ax2.set_ylabel("Amplitudo")
    ax2.legend()

    plt.pause(0.001)

    # Tampilkan frame video
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()