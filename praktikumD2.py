import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import math

def calculate_angle(A, B, C):
    """
    Hitung sudut di titik B menggunakan rumus dot product:
    θ = arccos( (A-B)·(C-B) / (||A-B|| * ||C-B||) )
    """
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    C = np.array(C, dtype=np.float32)

    BA = A - B
    BC = C - B

    dot_product = np.dot(BA, BC)
    magnitude = np.linalg.norm(BA) * np.linalg.norm(BC)

    if magnitude == 0:
        return 0

    cos_theta = dot_product / magnitude
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = math.degrees(math.acos(cos_theta))
    return angle

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5,
                        trackCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    if lmList:

        # Titik referensi berdasarkan MediaPipe
        hip = lmList[24][0:2]      # Pinggul kanan
        knee = lmList[26][0:2]     # Lutut kanan
        ankle = lmList[28][0:2]    # Pergelangan kaki kanan

        # Visualisasi titik referensi
        pts = [hip, knee, ankle]
        for pt in pts:
            cv2.circle(img, pt, 8, (0, 255, 255), cv2.FILLED)

        # Hitung sudut lutut
        angle_knee = calculate_angle(hip, knee, ankle)

        # Tampilkan sudut pada layar
        cv2.putText(img, f"Angle: {int(angle_knee)} deg",
                    (knee[0] - 60, knee[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Klasifikasi sederhana: berdiri atau jongkok
        status = "Standing" if angle_knee > 150 else "Squatting"
        cv2.putText(img, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        print("Knee Angle:", int(angle_knee), "| Status:", status)

    cv2.imshow("Pose + Angle", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()