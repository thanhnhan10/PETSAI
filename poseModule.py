import cv2
import mediapipe as mp
import math


class PoseDetector:
    
    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils #ve len khung xuong
        
#================khoi tao mediapipe===========

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

# ==================ve theo tu the==================

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #----------chuyen he mau tu webcam-----------
        self.results = self.pose.process(imgRGB)# ---------------ket qua nhan dien duoc luu vao result---------
        if self.results.pose_landmarks:# neu nhu phat hien khung xuong nguoi thi moi thuc hien nguoc lai khong
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img, self.results
    
#==================tim kiem danh sach thong so=====================

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape #------------co duoc hinh dang khung ban dau-------------
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                
# ==============ve khung xuong len anh===============

            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
# ================tim goc giua 3 diem bên tay trai==================

    def findAngle_left(self, img, p1, p2, p3, draw=True,toado_dung=180):
        # -----nhap gia tri mediapipe------
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
           
        if draw:
            # ==================ve duong line=======================
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)
            # ==================ve nut=======================
            cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (0, 255, 0), cv2.FILLED)
            # ==================ve vong bao nut=======================
            cv2.circle(img, (x1, y1), 8, (0, 0, 0), 2) 
            cv2.circle(img, (x2, y2), 8, (0, 0, 0), 2)
            cv2.circle(img, (x3, y3), 8, (0, 0, 0), 2)
            # ==================ve toa do=======================
            cv2.putText(img, str(int(angle)), (x2 + 10, y2 - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),1)
            # ==================ve diem=======================
            score_left = Score(int (angle),toado_dung,1,10,10) # ham truyen diem
            cv2.putText(img, str(score_left), (x2 +20, y2 -20),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        return angle,  score_left


#==============tim goc giua 3 diem bên tay phải===================

    def findAngle_right(self, img, p1, p2, p3, draw=True,toado_dung=180):
        # nhap gia tri mediapipe
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle/ 2:
            angle=360-angle

        if draw:
            # ==================ve duong line phai=======================
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)
           # ==================ve nut phải=======================
            cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (0, 255, 0), cv2.FILLED)
            # ==================ve vong bao nut phai=======================
            cv2.circle(img, (x1, y1), 8, (0, 0, 0), 2) 
            cv2.circle(img, (x2, y2), 8, (0, 0, 0), 2)
            cv2.circle(img, (x3, y3), 8, (0, 0, 0), 2)
            # ==================ve toa do phai=======================
            cv2.putText(img, str(int(angle)), (x2 -40, y2 - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            
            # #tinhdiem
            # if str(p2) in ds_diem_angle_right.keys():
            #     gocdung=ds_diem_angle_right[str(p2)]
            # else:
            #     gocdung=180
            # cv2.putText(img, "({0} - {1} - {2} - {3})".format(str(p1),str(p2),str(p3),str(gocdung)), (x2 - 70, y2 + 90),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            
            # ==================ve diem phai=======================
            score_right = Score(int (angle),toado_dung,1,5,10) # ham truyen diem
            cv2.putText(img, str(score_right), (x2 - 50, y2 -20),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                        
            return angle, score_right
    
# ====================tinh diem===================================
def Score (goc_that, goc_dung, muc_diem_tru=1, bien_do_lech=5, diem_ban_dau=10):
    do_lech=abs(goc_dung-goc_that)
    tylelech=do_lech/bien_do_lech
    diemtru=tylelech*muc_diem_tru
    diem=diem_ban_dau-round(diemtru,1)
    if diem<0:
        return 0
    lamtrondiem= diem- int(diem)
    if lamtrondiem>=0.5:
        diem=int(diem)+1
    if lamtrondiem>0 and lamtrondiem<0.5:
        diem=int(diem)+0.5

    return diem