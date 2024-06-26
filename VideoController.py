import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui

#########################################
wCam, hCam = 640, 480
#########################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
curTime = 0
prevTime = 0

detector = htm.handDetector(detectionCon=0.5)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minVol, maxVol = volRange[0], volRange[1]
volBar = 400
volPer = 0

# State control for pausing
thumb_touching_index = False
space_pressed = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # Index and Thumb
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3, y3, x4, y4, x5, y5 = lmList[12][1], lmList[12][2], lmList[16][1], lmList[16][2], lmList[20][1], lmList[20][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x3,y3), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x4,y4), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x5,y5), 15, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2), (255,0,255), 3)
        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # Volume range -54, -4
        # Hand Range 20, 220

        vol = np.interp(length, [20,200], [minVol, maxVol])
        volBar = np.interp(length, [20,200], [400, 150])
        volPer = np.interp(length, [20,200], [0, 100])
        print(length, vol)

        if (lmList[8][2] > lmList[5][2]) and (lmList[12][2] > lmList[9][2]) and (lmList[16][2] > lmList[13][2]) and (lmList[20][2] > lmList[17][2]):
            # test // cv2.circle(img, (50,50), 15, (255,255,255), cv2.FILLED)
            # TIPS below PALM + direction of thumb -> Fast-Foward or Roll-Back
            if lmList[4][1] < lmList[3][1] and lmList[4][1] < lmList[8][1]:
                # Fast-forward
                pyautogui.press('right')
            elif lmList[4][1] > lmList[3][1] and lmList[4][1] > lmList[8][1]:
                # Roll-back
                pyautogui.press('left')

        if length < 30 and all(y < y2 for y in [y3, y4, y5]) and not thumb_touching_index:
            # THUMB_TIP touches INDEX_TIP -> Pause
            cv2.circle(img, (cx, cy), 15 , (0, 255, 0), cv2.FILLED)
            thumb_touching_index = True

            # Simulate press spacebar
            if not space_pressed:
                pyautogui.press('space')
                space_pressed = True

        elif length >= 30:
            thumb_touching_index = False
            space_pressed = False

    # Show Volume
    cv2.rectangle(img, (50,150), (85,400), (255, 0, 0), 3)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Show FPS
    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime

    # Show Image
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Img", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
