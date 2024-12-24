import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Fungsi untuk membuat filter low-pass untuk menghaluskan sinyal rPPG
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Fungsi untuk menerapkan filter low-pass pada sinyal
def butter_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

# Variabel untuk sinyal respirasi dan rPPG
respiration_signal = []
rPPG_signal = []

# Parameter untuk pemrosesan sinyal
respiration_freq = 0.2  # frekuensi pernapasan dalam Hz
rPPG_freq = 1.5  # frekuensi detak jantung dalam Hz
cutoff_freq = 1.0  # frekuensi cutoff filter untuk rPPG (Hz)

# Membuat plot untuk visualisasi
plt.ion()  # interactive mode untuk pembaruan grafik
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
x_axis = np.arange(0, 100, 1)

# Fungsi untuk menghitung sinyal rPPG
def calculate_rPPG(frame):
    # Ubah gambar ke ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ambil nilai saluran "V" (nilai kecerahan) untuk sinyal PPG
    v_channel = hsv[:, :, 2]
    
    # Hitung rata-rata perubahan nilai V sebagai sinyal PPG
    avg_brightness = np.mean(v_channel)
    return avg_brightness

# Fungsi untuk menghitung sinyal respirasi
def calculate_respiration_signal(frame):
    # Menggunakan pergerakan area dada atau perut (misalnya dengan deteksi wajah)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Ambil area wajah (kita anggap sebagai bagian tubuh yang bergerak)
        (x, y, w, h) = faces[0]
        roi = frame[y:y+h, x:x+w]
        
        # Gunakan perbedaan dalam rata-rata nilai warna untuk menunjukkan pergerakan (respirasi)
        avg_color = np.mean(roi, axis=(0, 1))
        return avg_color[1]  # Menggunakan saluran G (hijau)
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
    
    # Filter sinyal rPPG
    if len(rPPG_signal) > 1:
        rPPG_signal_filtered = butter_filter(rPPG_signal, cutoff_freq, fps)
    
    # Visualisasikan sinyal
    ax1.clear()
    ax1.plot(x_axis[:len(rPPG_signal)], rPPG_signal_filtered[:len(rPPG_signal)], label="rPPG Signal")
    ax1.set_title("Sinyal rPPG")
    ax1.set_xlabel("Waktu (detik)")
    ax1.set_ylabel("Amplitudo")
    ax1.legend()

    ax2.clear()
    ax2.plot(x_axis[:len(respiration_signal)], respiration_signal[:len(respiration_signal)], label="Sinyal Respirasi")
    ax2.set_title("Sinyal Respirasi")
    ax2.set_xlabel("Waktu (detik)")
    ax2.set_ylabel("Amplitudo")
    ax2.legend()

    plt.pause(0.001)  # Pause untuk pembaruan grafik

    # Tampilkan frame video dengan deteksi wajah (opsional)
    cv2.imshow("Webcam", frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup jendela video dan grafik
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Matplotlib berhenti dalam mode interaktif
plt.show()
