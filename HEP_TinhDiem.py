import cv2
import numpy as np
import poseModule as pm


def KiemTraVuotManHinh(width,height):
   if width<=1300 and height<=750:
       return 1
   return 0

ds_diemdung_DT={
               'VUONTHO_BD_KT_NHIP_2':{
                "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'VUONTHO_NHIP_1_NHIP_3':{
                    "11":160,"13":163,"12":160,"14":163,"23":183,"25":175,"24":183,"26":175
                },
                'TAY_BD_KT':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                 'TAY_NHIP_1':{
                    "11":95,"13":170,"12":95,"14":170,"23":167,"25":180,"24":167,"26":180
                },
                'TAY_NHIP_2':{
                    "11":175,"13":172,"12":175,"14":172,"23":170,"25":178,"24":170,"26":178
                },
                'TAY_NHIP_3':{
                    "11":90,"13":172,"12":90,"14":172,"23":186,"25":182,"24":186,"26":182
                },
                'CHAN_BD_KT':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'CHAN_NHIP_1_NHIP_3':{
                    "11":90,"13":172,"12":90,"14":172,"23":186,"25":182,"24":186,"26":182
                },
                'CHAN_NHIP_2':{
                    "11":75,"13":163,"12":75,"14":163,"23":175,"25":207,"24":175,"26":207
                },
                'LUON_BD_KT':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'LUON_NHIP_1':{
                    "11":175,"13":172,"12":175,"14":172,"23":170,"25":178,"24":170,"26":178
                },
                'LUON_NHIP_2':{
                    "11":214,"13":180,"12":214,"14":180,"23":200,"25":183,"24":200,"26":183
                },
                'LUON_NHIP_3':{
                    "11":90,"13":172,"12":90,"14":172,"23":186,"25":182,"24":186,"26":182
                },
                'BUNG_BD_KT':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'BUNG_NHIP_1' :{
                    "11":165,"13":178,"12":165,"14":178,"23":168,"25":178,"24":168,"26":178
                },
                'BUNG_NHIP_2':{
                    "11":85,"13":180,"12":85,"14":180,"23":180,"25":180,"24":180,"26":180
                },
                'BUNG_NHIP_3':{
                    "11":90,"13":172,"12":90,"14":172,"23":186,"25":182,"24":186,"26":182
                },
                'TOANTHAN_BD_KT':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'TOANTHAN_NHIP_1':{
                    "11":160,"13":163,"12":160,"14":163,"23":183,"25":175,"24":183,"26":175
                },
                'TOANTHAN_NHIP_2':{
                    "11":19,"13":180,"12":18,"14":180,"23":84,"25":200,"24":84,"26":200
                },
                'TOANTHAN_NHIP_3':{
                    "11":87,"13":184,"12":87,"14":184,"23":174,"25":180,"24":174,"26":180
                },
                'NHAY_BD_KT_NHIP_2':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'NHAY_NHIP_1':{
                    "11":90,"13":182,"12":90,"14":182,"23":166,"25":180,"24":166,"26":180
                },
                'NHAY_NHIP_3':{
                    "11":180,"13":163,"12":180,"14":163,"23":165,"25":180,"24":165,"26":180
                },
                'DIEUHOA_BD_KT':{
                    "11":18, "13":190, "12":18, "14":190, "23":175, "25":184, "24":175, "26":184
                },
                'DIEUHOA_NHIP_1_NHIP_3':{
                    "11":90,"13":185,"12":90,"14":185,"23":160,"25":230,"24":160,"26":230
                },
                'DIEUHOA_NHIP_2':{
                    "11":12,"13":215,"12":12,"14":215,"23":178,"25":180,"24":178,"26":180
                },
                "CHUA_PHAT_HIEN_DT":{
                     "11":360,"13":360,"12":360,"14":360,"23":360,"25":360,"24":360,"26":360
                }
                
            }
detector=pm.PoseDetector()

def Timboxuongnguoi(img):
    img1, results=detector.findPose(img)
    Top_X, Top_Y, Bottom_X, Bottom_Y = CatHinhNguoiToanThan(img, results)
    img1 = cv2.rectangle(img1, (Top_X,Top_Y), (Bottom_X,Bottom_Y), (255,0,0), 2)
    roi = img[Top_Y:Bottom_Y, Top_X:Bottom_X]
    # cv2.imshow("roi",roi)
#     cv2.imshow("camera", img1)
    return img1, roi

def TinhDiem(img1, dongtac):
    global left_tay
    lmList=detector.findPosition(img1,False)
    if len (lmList)!=0:
       score_right_13=0
       score_right_14=0
       score_right_25=0
       score_right_23=0
       score_right_11=0
       score_right_12=0
       score_right_26=0
       score_right_24=0
       angle=0
       if KiemTraVuotManHinh(lmList[13][1],lmList[13][2]):
            angle, score_right_13= detector.findAngle_left(img1, 11, 13, 15,ds_diemdung_DT[str(dongtac)][str("13")])
       if KiemTraVuotManHinh(lmList[14][1],lmList[14][2]):
            angle, score_right_14= detector.findAngle_right(img1, 12, 14, 16,ds_diemdung_DT[str(dongtac)][str("14")])
       if KiemTraVuotManHinh(lmList[23][1],lmList[23][2]):
            angle, score_right_23= detector.findAngle_left(img1, 11, 23, 25,ds_diemdung_DT[str(dongtac)][str("23")])
       if KiemTraVuotManHinh(lmList[24][1],lmList[24][2]):
            angle, score_right_24= detector.findAngle_right(img1, 12, 24, 26,ds_diemdung_DT[str(dongtac)][str("24")])
       if KiemTraVuotManHinh(lmList[26][1],lmList[26][2]):
            angle, score_right_26= detector.findAngle_right(img1, 24, 26, 28,ds_diemdung_DT[str(dongtac)][str("26")])
       if KiemTraVuotManHinh(lmList[25][1],lmList[25][2]):
            angle, score_right_25= detector.findAngle_left(img1, 23, 25, 27,ds_diemdung_DT[str(dongtac)][str("25")])
       if KiemTraVuotManHinh(lmList[12][1],lmList[12][2]):
            angle, score_right_12= detector.findAngle_right(img1, 14, 12, 24,ds_diemdung_DT[str(dongtac)][str("12")])
       if KiemTraVuotManHinh(lmList[11][1],lmList[11][2]):
            angle, score_right_11= detector.findAngle_left(img1, 13, 11, 23,ds_diemdung_DT[str(dongtac)][str("11")])
       
       
       left_tay=round(np.average([score_right_11,score_right_13]), 1)
       right_tay=round(np.average([score_right_12,score_right_14]), 1)
       left_chan=round(np.average([score_right_23,score_right_25]), 1) 
       right_chan=round(np.average([score_right_24,score_right_26]), 1)

       tb_toanthan=round(np.average([left_chan,left_tay,right_chan,right_tay]), 1) 

       cv2.putText(img1, "Tay trai={0}".format(str(left_tay)), (20,20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1) 
       cv2.putText(img1, "Tay phai={0}".format(str(right_tay)), (20,45),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1) 
       cv2.putText(img1, "chan trai={0}".format(str(left_chan)), (20,70),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1) 
       cv2.putText(img1, "chan phai={0}".format(str(right_chan)), (20,95),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
       cv2.putText(img1, "DIEM SO={0}".format(str(tb_toanthan)), (20,120),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    
       cv2.putText(img1, "Nhip {0}".format(str(dongtac)), (230,45),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    return img1, tb_toanthan


def CatHinhNguoiToanThan(image, results):
     image_height, image_width, _ = image.shape
     Mui = int(results.pose_landmarks.landmark[0].y * image_height)
     CP = int(results.pose_landmarks.landmark[27].y * image_height)
     TT = int(results.pose_landmarks.landmark[16].x * image_width)
     TP = int(results.pose_landmarks.landmark[15].x * image_width)
     TT_Y = int(results.pose_landmarks.landmark[16].y * image_height)
     TP_Y = int(results.pose_landmarks.landmark[15].y * image_height)
     CP_X = int(results.pose_landmarks.landmark[27].x * image_width)
     CT_X = int(results.pose_landmarks.landmark[26].x * image_width)
      
     TangThemX = 55
     TangThemY = 60
     
     Top_X=TT
     Top_Y = Mui
     Bottom_X = TP
     Bottom_Y=CP
     if Top_X > Bottom_X:
          Bottom_X = TT
          Top_X=TP
     if TP_Y < Mui or TT_Y<Mui:
          if TP_Y<TT_Y:
               Top_Y = TP_Y
          else:
               Top_Y = TT_Y
     if CT_X<TT:
          Top_X=CT_X
     if CP_X > TP:
          Bottom_X = CP_X
      
        

     Top_Y = Top_Y - TangThemY
     Bottom_Y = Bottom_Y +TangThemY
     Top_X = Top_X - TangThemX
     Bottom_X = Bottom_X + TangThemX
      
     if Top_X<=0:
          Top_X=10
      
     if Top_Y<=0:
          Top_Y=10
     return Top_X, Top_Y, Bottom_X, Bottom_Y