# ========================================================THU VIEN==========================================================
from tkinter import *
from tkinter import ttk
import tkinter 
#----------------------------------
import cv2
import PIL.Image,PIL.ImageTk
import HEP_TinhDiem as hep
import threading
#----------------------------------
import os
import playsound
import speech_recognition as sr
import time
import sys
import ctypes
import datetime
import json
import re
import urllib
import urllib.request as urllib2
import random
from gtts import gTTS
from tkinter import filedialog
from tkinter import HORIZONTAL
# --------------------------------------------
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
import tensorflow  as tf
from keras.preprocessing import image
from numpy import argmax
import customtkinter
# -----------------------------------------------

language = 'vi'

# ========================================================LOGO_GIAODIEN==========================================================

window= Tk()
window.title("Hệ thống hỗ trợ chấm điểm GDTC")# tieu de
# icon tieu de
photo = PhotoImage(file = 'icon/ctu.png')
window.iconphoto(False, photo)
window.geometry('670x680+850+80')#kich thuoc mang hinh va vi tri xuat hien
window.configure(bg="#BACDDB")

# ========================================================GOI VideoCapture==========================================================

cap=cv2.VideoCapture("./video_bung.mp4")

# ========================================================HAM XU LY==========================================================

def openf():
    return filedialog.askopenfilename()

#----------------------------------------------- 

def xulycamera():
    global cap
    cap=cv2.VideoCapture(0)
    XLHA()
    
#-----------------------------------------------

def xulyfile():
    global cap
    cap=cv2.VideoCapture(openf())
    XLHA()
    
#-----------------------------------------------

def TrungVi(array):
    if len(array) < 1:
        return(None)
    if len(array) % 2 == 0:
        median = (array[len(array)//2-1: len(array)//2+1])
        return sum(median) / len(median)
    else:
        return(array[len(array)//2])
    
# ---------------------------------------------- 
   
def ktra_ds( DSKT, text):
    for i in DSKT:
        if i in text:
            return True
    return False

#------------------------------------------

def checkCategories(dictDT):
    listCate = dictDT.values()
    return sum(listCate)

# ---------------------------------------------- 
  
def speak(text1):
    r1 = random.randint(1,10000000)
    r2 = random.randint(1,10000000)

    randfile = str(r2)+"randomtext"+str(r1) +".mp3"

    tts = gTTS(text=text1, lang='vi', slow=True)
    tts.save(randfile)
    print(randfile)
    playsound.playsound(randfile)
    print(randfile)
    os.remove(randfile)
    
#-----------------------------------------------

def majority_element(num_list):
        idx, ctr = 0, 1
        
        for i in range(1, len(num_list)):
            if num_list[idx] == num_list[i]:
                ctr += 1
            else:
                ctr -= 1
                if ctr == 0:
                    idx = i
                    ctr = 1
        
        return num_list[idx]    
    
# ---------------------------------------------- 

def GetDictionary_DongTac(_Categories):
    d={}
    for each in _Categories:
        d[each]=0
    return d

# --------------------------------------------------------------

model=keras.models.load_model("models\\vtho1000.h5", compile=False)
categories = ['VUONTHO_BD_KT_NHIP_2', 'VUONTHO_NHIP_1_NHIP_3', 'CHUA_PHAT_HIEN_DT']

# ---------------------------------------------- 

KetQuaToanDongTac={} 
dongtac="CHUA_PHAT_HIEN_DT"

def frame():
    global  stop_threads_audio
    global Diemhienthi
    Dem=0
    listOfNhip = []
    KiemTraThayDoi=""
    DiemSoNgauNhien_TungDongTac=[]
    nhip_Nhan="CHUA_PHAT_HIEN_DT"
    ToanDongTac_HienHanh= GetDictionary_DongTac(categoriesVUONTHO)
    while (cap.isOpened()):
        try:
            ret , img=cap.read()
    
            if dongtac in ["Dong tac Vuon tho","Dong tac Tay","Dong tac Chan","Dong tac Luon","Dong tac Bung","Dong tac Toan than","Dong tac Nhay","Dong tac Dieu hoa"]: 
                img, anhcat_roi = hep.Timboxuongnguoi(img) 
                image_nhandang = cv2.resize(anhcat_roi, (32, 32))  
                image_nhandang = np.array(image_nhandang, dtype="float") / 255.0            
                image_nhandang=np.expand_dims(image_nhandang, axis=0)
                pred=model.predict(image_nhandang)
                Res=argmax(pred,axis=1)             
                nhip=categories[Res[0]]
                listOfNhip.append(nhip)
                #--------------------------------------------
                if len(listOfNhip)>10:
                    nhip_Nhan=majority_element(listOfNhip)
                    # print(listOfNhip)
                    print(nhip_Nhan)
                    listOfNhip=[]
                img,tb_toanthan= hep.TinhDiem(img,nhip_Nhan)
                #-------------------------------------------------
                if KiemTraThayDoi!=nhip_Nhan:
                    Dem=0
                    if(len(DiemSoNgauNhien_TungDongTac)>0):
                        print("=======================================\n")
                        print("Diem:",DiemSoNgauNhien_TungDongTac)
                        DiemSoTV = TrungVi(DiemSoNgauNhien_TungDongTac)
                        
                        #---------Diem Dong Tac-----------
                        # DiemHienTai = KetQuaToanDongTac[KiemTraThayDoi]
                        # print("kiem tra1:",ToanDongTac_HienHanh)
                        # print("het qua dong tac 2:",KetQuaToanDongTac)
                        if ToanDongTac_HienHanh[KiemTraThayDoi]<DiemSoTV:
                            ToanDongTac_HienHanh[KiemTraThayDoi]=DiemSoTV
                            
                        #------------------------------
                        print("\n Trung vi",round(DiemSoTV,1))
                        DiemSoTB=np.mean(DiemSoNgauNhien_TungDongTac)
                        print("\n Trung binh:",round(DiemSoTB,1))
                        print("DS_DT: ",ToanDongTac_HienHanh)
                        DiemSoNgauNhien_TungDongTac=[]
                        Diem_DT=sum(ToanDongTac_HienHanh.values())/len(ToanDongTac_HienHanh)
                        print("Diem toan DT: ",round(Diem_DT,1))
                        #----------------------------------------------------
                        Diemhienthi=""
                        Diem=""
                        TenTVDT={"VUONTHO_BD_KT_NHIP_2":"Vươn thở, bắt đầu, nhịp 2",
                                 "NHAY_BD_KT_NHIP_2":"Nhảy bắt đầu, nhịp 2",
                                 "VUONTHO_NHIP_1_NHIP_3":"Vươn thở, bắt đầu, nhịp 1, nhịp 3",
                                 "DIEUHOA_NHIP_1_NHIP_3":"Điều hòa nhịp 1, nhịp 3",
                                 "TAY_BD_KT":"Tay bắt đầu, kết thúc",
                                 "CHAN_BD_KT":"Chân bắt đầu, kết thúc",
                                 "LUON_BD_KT":"Lườn bắt đầu, kết thúc",
                                 "BUNG_BD_KT":"Bụng bắt đầu, kết thúc",
                                 "TOANTHAN_BD_KT":"Toàn thân Bắt đầu, kết thúc",
                                 "DIEUHOA_BD_KT":" Điều hòa bắt đầu, kết thúc",
                                 "NHAY_NHIP_1":"Nhảy nhịp 1",
                                 "TAY_NHIP_1":"Tay nhịp 1",
                                 "CHAN_NHIP_1":"Chân nhịp 1",
                                 "LUON_NHIP_1":"Lườn nhịp 1",
                                 "BUNG_NHIP_1":"Bụng nhịp 1",
                                 "TOANTHAN_NHIP_1" :"Toàn thân nhịp 1",
                                 "TAY_NHIP_2":"Tay nhịp 2",
                                 "CHAN_NHIP_2":"Chân nhịp 2",
                                 "LUON_NHIP_2":"Lườn nhịp 2",
                                 "BUNG_NHIP_2":"Bụng nhịp 2",
                                 "TOANTHAN_NHIP_2":"Toàn thân nhịp 2",
                                 "NHAY_NHIP_2":"Nhảy nhịp 2",
                                 "DIEUHOA_NHIP_2":"Điều hòa nhịp 2",
                                 "TAY_NHIP_3":"Tay nhịp 3",
                                 "LUON_NHIP_3":"Lườn nhịp 3",
                                 "BUNG_NHIP_3":"Bụng nhịp 3",
                                 "TOANTHAN_NHIP_3":"Toàn thân nhịp 3",
                                 "NHAY_NHIP_3":"Nhảy nhịp 3",
                                 "CHUA_PHAT_HIEN_DT":"không phải động tác"}
                        
                        for dt,e in ToanDongTac_HienHanh.items():
                            Diemhienthi="{0}{1}: {2}\n".format(Diemhienthi,TenTVDT.get(dt),round(e,1))
                            
                        Diem_tb="{0}\nĐIỂM: {1}".format(Diem,round(Diem_DT,1))
                        if checkCategories(ToanDongTac_HienHanh)>0:
                            diemtungnhip.config(text = Diemhienthi)
                            diemtb.config(text = Diem_tb)
                        else:
                            diemtungnhip.config(text = "Dang thuc hien")
                            diemtb.config(text = "Dang thuc hien")
                    
                    KiemTraThayDoi=nhip_Nhan
                    
                else:
                    Dem+=1
                if Dem>SLFrame_min and Dem<SLFrame_max and Dem%5==0:
                    DiemSoNgauNhien_TungDongTac.append(tb_toanthan)
                tenDT.config(text =dongtac)
                cv2.putText(img, "{0}".format(str(dongtac)), (230,20),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)                 

            cv2.imshow("camera", img)
            # cv2.imshow("camera1", img)
            if cv2.waitKey(WK_camera) & 0xFF == ord('q'):
                stop_threads_audio=True
                break
            if stop_threads:                
                stop_threads_audio=True                
                break
        except:
            # dongtachienhanh = dongtac
            ToanDongTac_HienHanh = KetQuaToanDongTac.copy()  
            continue
   
# ---------------------------------------------- 

Diemhienthi=0
Diem=0
SLFrame_min=25
SLFrame_max=100
WK_camera=0

# ----------------------------------------

def XLHA():
    threadOfFrame = threading.Thread(target=frame)
    threadOfFrame.start() 
    global stop_threads
    stop_threads=False
    XLAT()
stop_threads=False     

# ---------------------------------------------- 
  
def XLAT():
    print("Am thanh hoat dong")
    threadOfFrame = threading.Thread(target=XuLyAmThanh)
    threadOfFrame.start()
    global stop_threads_audio
    stop_threads_audio=False
   
# ---------------------------------------------- 
  
from win32com.client import constants, Dispatch 
def asst_speaks(audio):  
    speaker = Dispatch("SAPI.SpVoice")
    speaker.speak(audio)    
    del speaker
   
# ---------------------------------------------- 
  
modelVtho=keras.models.load_model("models\\vtho1000.h5", compile=False)
modelTay=keras.models.load_model("models\\tay1000.h5", compile=False)
modelChan=keras.models.load_model("models\\chan1000.h5", compile=False)
modelLuon=keras.models.load_model("models\\luon1000.h5", compile=False)
modelBung=keras.models.load_model("models\\bung1000.h5", compile=False)
modelToanthan=keras.models.load_model("models\\toanthan1000.h5", compile=False)
modelNhay=keras.models.load_model("models\\nhay1000.h5", compile=False)
modelDieuhoa=keras.models.load_model("models\\dieuhoa1000.h5", compile=False)
   
# ---------------------------------------------- 
  
categoriesVUONTHO = ['VUONTHO_BD_KT_NHIP_2', 'CHUA_PHAT_HIEN_DT', 'VUONTHO_NHIP_1_NHIP_3']
categoriesTAY = ['TAY_BD_KT', 'CHUA_PHAT_HIEN_DT', 'TAY_NHIP_1', 'TAY_NHIP_2','TAY_NHIP_3']
categoriesCHAN = ['CHAN_BD_KT', 'CHAN_NHIP_1_NHIP_3', 'CHAN_NHIP_2','CHUA_PHAT_HIEN_DT']
categoriesLUON = ['LUON_BD_KT','CHUA_PHAT_HIEN_DT','LUON_NHIP_1', 'LUON_NHIP_2', 'LUON_NHIP_3']
categoriesBUNG = ['BUNG_BD_KT','BUNG_NHIP_1', 'BUNG_NHIP_2', 'BUNG_NHIP_3','CHUA_PHAT_HIEN_DT']
categoriesTOANTHAN = ['TOANTHAN_BD_KT','CHUA_PHAT_HIEN_DT','TOANTHAN_NHIP_1', 'TOANTHAN_NHIP_2', 'TOANTHAN_NHIP_3']
categoriesNHAY = ['NHAY_BD_KT_NHIP_2', 'CHUA_PHAT_HIEN_DT','NHAY_NHIP_1', 'NHAY_NHIP_3' ]
categoriesDIEUHOA = ['DIEUHOA_BD_KT', 'DIEUHOA_NHIP_1_NHIP_3', 'DIEUHOA_NHIP_2', 'CHUA_PHAT_HIEN_DT']
   
# ---------------------------------------------- 
  
import pyttsx3
def XuLyAmThanh():
    rec = sr.Recognizer()
    global dongtac
    global model
    global categories
    global KetQuaToanDongTac
    engine = pyttsx3.init()
    while(cap.isOpened()): 
        try: 
            
    #-----------------------------------------------------------------
            with sr.Microphone() as mic:
                audio = rec.listen(mic, phrase_time_limit=5)
            
                text = rec.recognize_google(audio, language="vi-VN")
                # print(text)
                if len(text)>0:
                    KTDS=["bắt đầu","ban đầu","thực hiện", "đầu", "hiện","đau"]
                    if ktra_ds(KTDS,text)==True: 
                        kiemtra=False
                        DSKT=["Phương Thảo","động tác Vương Thẩm","Phương Phở","Vương thở","vương tử","vươn thở","thở",
                            'động tác Vương thể','cộng tác viên phải','động tác Vương ở','cộng tác Vương cờ','động tác Phương Thảo',
                            'động tác Khương Thảo','Hương Thảo','cộng tác viên thảo','cộng tác viên thể', 'vươn']
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                                                    
                                dongtac = "Dong tac Vuon tho"
                                speak("Bạn đang chọn động tác vươn thở")
                                model=modelVtho
                                categories = categoriesVUONTHO
                                KetQuaToanDongTac=GetDictionary_DongTac(categories)   
                                engine.runAndWait()
                                kiemtra=True 
                                
                        # ----------------------------------------------------
                            
                        DSKT=["cộng tác tay","tay",'động tác tăng','tây','tang']
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                                  
                                dongtac = "Dong tac Tay"                          
                                speak("Bạn đang chọn động tác tay")
                                model=modelTay
                                categories = categoriesTAY
                                KetQuaToanDongTac=GetDictionary_DongTac(categories) 
                                engine.runAndWait()                                  
                                kiemtra=True 
                                
                        # ----------------------------------------------------
                        
                        DSKT=["động tác chăn","động tác Trăng","cộng tác chăn","Sóc Trăng","chân", "động tác chân","chăn"]
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                                         
                                dongtac = "Dong tac Chan"                            
                                speak("Bạn đang chọn động tác chân")
                                model=modelChan
                                categories = categoriesCHAN
                                KetQuaToanDongTac=GetDictionary_DongTac(categories) 
                                engine.runAndWait() 
                                kiemtra=True 

                        # ----------------------------------------------------
                        
                        DSKT=["động tác lừa","động tác lường","Cát Lượng","lườn",'tổng chất lượng','tổng tắc lượng',
                            'tóm tắt lượng','động tác làm','động tác lưu','tổng các lượng','động tác luôn', 'lươn','thực hiện lười',
                            'thực hiện động tác đường','thực hiện động tác giường','bắt đầu lồn','thực hiện động tác lường','thực hiện động tác thường',
                            'thực hiện lời','thực hiện đường','thực hiện lợn','thiện lương','bắt đầu đường','bắt đầu giường','bắt đầu lường',
                            'bắt đầu thường','đầu lợn','đào động tác Lượng','thực hiện lười','thực hiện động tác trườn','cải lương','thực hiện động tác trườn'
                            'bắt đầu lân','bắt đầu động tác nào', 'câu lươn','màu động tác lường','tóm tắt lương','chuyển động tác lương']
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                                      
                                dongtac = "Dong tac Luon" 
                                speak("Bạn đang chọn động tác lườn")
                                model=modelLuon
                                categories = categoriesLUON
                                KetQuaToanDongTac=GetDictionary_DongTac(categories) 
                                engine.runAndWait()  # bien cho  
                                kiemtra=True 
                            
                        # ----------------------------------------------------
                        
                        DSKT=["bung","bụng",'cộng tác dụng','cộng tác bụng','tác dụng',"động tác bụng"]
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                                      
                                dongtac = "Dong tac Bung"
                                speak("Bạn đang chọn động tác bụng")  
                                model=modelBung 
                                categories = categoriesBUNG
                                KetQuaToanDongTac=GetDictionary_DongTac(categories)  
                                engine.runAndWait() 
                                kiemtra=True 
                            
                        # ----------------------------------------------------
                        
                        DSKT=["toàn thân","toan thân", "động tác toàn thân",'thân','toàn']
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                                       
                                dongtac = "Dong tac Toan than"
                                speak("Bạn đang chọn động tác toàn thân")  
                                model=modelToanthan
                                categories = categoriesTOANTHAN 
                                KetQuaToanDongTac=GetDictionary_DongTac(categories) 
                                engine.runAndWait() 
                                kiemtra=True 
                                
                        # ----------------------------------------------------
                        
                        DSKT=["nhảy", "nhay","động tác nhảy", 'nhãy']
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:                              
                                dongtac = "Dong tac Nhay"
                                speak("Bạn đang chọn động tác nhảy") 
                                model=modelNhay
                                categories = categoriesNHAY  
                                KetQuaToanDongTac=GetDictionary_DongTac(categories)
                                engine.runAndWait() 
                                kiemtra=True 
                                
                        # ----------------------------------------------------
                        
                        DSKT=["điều hòa","dieu hòa","hòa", "động tác điều hòa", 'hòa']
                        if kiemtra==False:
                            if ktra_ds(DSKT,text)==True:            
                                dongtac = "Dong tac Dieu hoa"
                                speak("Bạn đang chọn động tác điều hòa")
                                model=modelDieuhoa
                                categories = categoriesDIEUHOA
                                KetQuaToanDongTac=GetDictionary_DongTac(categories)
                                engine.runAndWait() 
                                kiemtra=True    
                                                                     
                        # ----------------------------------------------------
                        
                        DSKT=[""]
                        if kiemtra==False:
                            speak("Bạn vui lòng nói đúng tên động tác")
                    DSKT=["rời khỏi","thoát khỏi","kết thúc","xong","Đóng","đóng","Thoát"]
                    if ktra_ds(DSKT,text)==True:
                        speak("Bạn đang chọn thoát chương trình")
                        dongtac="CHUA_PHAT_HIEN_DT"
                        speak("Xin cảm ơn")
                        # break 
                    if stop_threads_audio:
                        break      
        except:                
            if stop_threads_audio:
                    break 
            continue
stop_threads_audio=False
   
# ---------------------------------------------- 
  
def tat():
    global stop_threads
    stop_threads=True
    global  stop_threads_audio
    stop_threads_audio=True
    cap.release()
    
# ----------------------------------------------

def ExitWindow():
    global stop_threads
    stop_threads = True  
    window.destroy()

# ========================================================GIAO DIEN==========================================================

w=643
h=647

vien_chung=Canvas(window, bg="#BBBBBB",width=w, height=h)
vien_chung.place(relx=0.02, rely=0.01)
vien_chung.create_rectangle(3,3,w,h, outline="black",width=1.5)

w_KQ=320
h_KQ=290

vien_KQ=Canvas(window, bg='#FFFFFF',width=w_KQ, height=h_KQ, borderwidth=3,bd=3, relief="sunken")
vien_KQ.place(relx=0.38, rely=0.36)
vien_KQ.create_rectangle(3,3,w_KQ+4,h_KQ+4)

# ===============================================================

he_thong=Label(window, text="HỆ THỐNG HỖ TRỢ CHẤM ĐIỂM \n MÔN GIÁO DỤC THỂ CHẤT",
               font = ("Times New Roman", 14, "bold"),
               fg='#362FD9',
               bg="#BBBBBB" )#, borderwidth=0.5, relief="raised")#394867
he_thong.place(relx=0.5,rely=0.07, anchor=N)

he_thong=Label(window, text="-------------o00o-------------",
               font = ("Times New Roman", 10, "bold"),
               fg='black',
               bg="#BBBBBB")
he_thong.place(relx=0.5,rely=0.15, anchor=N)

# ========================================================

label_DC=Label(window, text = "Thanh điều chỉnh.",
          font = ("Times New Roman", 14 ,"bold"),
          fg='Maroon',
          bg="#BBBBBB")
label_DC.place(relx=0.05,rely=0.4, anchor=SW)

# ------------------------

label_min_slframe=Label(window, text = "Frame bắt đầu tính điểm:",
          font = ("Times New Roman", 11 ,"bold"),
          fg='#222222',
          bg="#BBBBBB")
label_min_slframe.place(relx=0.05,rely=0.43, anchor=SW)

def cmd_SLMin(newVal):
    global SLFrame_min
    gt_min=int(float(newVal))
    SLFrame_min=gt_min
    label_SLframe_min["text"]=gt_min
   
label_SLframe_min=Label()#khi can hien thi label_SLframe_min.place(relx=0.07, rely=0.8, anchor=NW)
scalemin=Scale(window,relief="sunken",bd=3,	highlightbackground="black",troughcolor="#729FB0",
               font = ("Times New Roman", 9 ,"bold"),
               fg='blue', 
               orient=HORIZONTAL,
               showvalue=2,
               width=10,
               sliderlength=20, 
               length=162,
               from_=25,to=100,
               command=cmd_SLMin)
scalemin.place(relx=0.05,rely=0.50, anchor=SW)

# ------------------------------------------

label_max_slframe=Label(window, text = "Frame kết thúc tính điểm:",
          font = ("Times New Roman", 11 ,"bold"),
          fg='#222222',
          bg="#BBBBBB")
label_max_slframe.place(relx=0.05,rely=0.54, anchor=SW)

def cmd_SLMax(newVal):
    global SLFrame_max
    gt_max=int(float(newVal))
    SLFrame_max=gt_max
    label_SLframe_max["text"]=gt_max
   
label_SLframe_max=Label()# 
scalemax=Scale(window, relief="sunken",bd=3,highlightbackground="black",troughcolor="#729FB0",
               font = ("Times New Roman", 9 ,"bold"),
               fg='blue',
               orient=HORIZONTAL,
               showvalue=2,
               width=10, 
               sliderlength=20, 
               length=160,
               from_=100,
               to=200,
               command=cmd_SLMax)
scalemax.place(relx=0.05,rely=0.61, anchor=SW)

# --------------------------------------------

tocdocamera=Label(window, text = "Tốc độ camera:",
          font = ("Times New Roman", 11 ,"bold"),
          fg='#222222',
          bg="#BBBBBB")
tocdocamera.place(relx=0.05,rely=0.65, anchor=SW)

def WK_camera(newVal):
    global WK_camera
    gt_WK_camera=int(float(newVal))
    WK_camera=gt_WK_camera
    label_WK_camera["text"]=gt_WK_camera
   
label_WK_camera=Label()# 
scalemax=Scale(window, relief="sunken",bd=3,highlightbackground="black",troughcolor="#729FB0",
               font = ("Times New Roman", 9 ,"bold"),
               fg='blue', 
               orient=HORIZONTAL,
               showvalue=2,
               width=10, 
               sliderlength=20,
               length=160,
               from_=0,
               to=50,
               command=WK_camera)
scalemax.place(relx=0.05,rely=0.72, anchor=SW)

#===================================================================================

ketqua=Label(window, text = "Kết quả:",
          font = ("Times New Roman", 18, "bold"),
          fg='#CF0A0A',
          bg="#BBBBBB")# borderwidth=0.5, relief="raised")
ketqua.place(relx=0.55, rely=0.28, anchor=NW)

# -----------------------

label_tenDT=Label(window, text = "Tên động tác:",
          font = ("Times New Roman", 13, "bold"),
          fg='#3f609a',
          bg='#FFFFFF')
label_tenDT.place(relx=0.45, rely=0.38, anchor=NW)

# -------------------------

tenDT=Label(window, text ="Chưa phát hiện",
          font = ("Times New Roman", 14),
          fg='red',
          bg='#FFFFFF')
tenDT.place(relx=0.45, rely=0.42, anchor=NW)

#-----------------------

ketquanhip=Label(window, text = "Điểm từng nhịp:",
          font = ("Times New Roman", 13, "bold"),
          fg='#3f609a',
          bg='#FFFFFF')
ketquanhip.place(relx=0.45, rely=0.46, anchor=NW)

# ---------------------

diemtungnhip=Label(window, text = "{0}".format(str("Chưa có điểm")),
          font = ("Times New Roman", 14),
          fg='red',
          bg='#FFFFFF')
diemtungnhip.place(relx=0.45, rely=0.5, anchor=NW)

# ---------------------

ketqua=Label(window, text = "Điểm TB động tác:",
          font = ("Times New Roman", 13, "bold"),
          fg='#3f609a',
          bg='#FFFFFF')
ketqua.place(relx=0.45, rely=0.67, anchor=NW)

# -------------------

diemtb=Label(window, text = "{0}".format(str(0)),
          font = ("Times New Roman", 14),
          fg='red',
          bg='#FFFFFF')
diemtb.place(relx=0.5, rely=0.7, anchor=NW)

# ============================================================

webcam=PIL.Image.open("icon/webcamv1.png")
resize_webcam=webcam.resize((20,20),PIL.Image.ANTIALIAS)
imgwebcam=PIL.ImageTk.PhotoImage(resize_webcam)
# nut button

button_cam=Button(window, text="Mở camera",width= "95",height="20",command=xulycamera,image=imgwebcam,compound = RIGHT, 
    font = ("Times New Roman", 13),
    bg='#4E9F3D',
    fg='white',
    borderwidth=2,
    bd=3,
    relief="raised",
    activebackground='#00abfd',
    activeforeground='white')
# button_cam.place()
button_cam.place(relx=0.16, rely=0.88, anchor=NW)

# ------------------

file=PIL.Image.open("icon/folder.png")
resize_file=file.resize((20,20),PIL.Image.ANTIALIAS)
imgfile=PIL.ImageTk.PhotoImage(resize_file)

buttonmofile=Button(window, text="Mở file ",width= "95",height="20",command=xulyfile,image=imgfile,compound = RIGHT, 
    font = ("Times New Roman", 13),
    bg='gray',
    fg='white',
    borderwidth=3,
    activebackground='#00abfd',
    activeforeground='white')
buttonmofile.place(relx=0.33, rely=0.88, anchor=NW)

#------------------

dong=PIL.Image.open("icon/close.png")
resize_dong=dong.resize((20,20),PIL.Image.ANTIALIAS)
imgdong=PIL.ImageTk.PhotoImage(resize_dong)

button_cam=Button(window, text="Tắt    ",width= "95",height="20",command=tat,image=imgdong,compound = RIGHT, 
    font = ("Times New Roman", 13),
    bg='white',
    fg='red',
    borderwidth=3,
    activebackground='#00abfd',
    activeforeground='white')
button_cam.place(relx=0.5, rely=0.88, anchor=NW)

#-----------------

button_tat=Button(window, text="Đóng  ",width= "95",height="20",command=ExitWindow,image=imgdong,compound = RIGHT,
    font = ("Times New Roman", 13),
    bg='red',
    fg='white',
    borderwidth=3,
    activebackground='#df322b',
    activeforeground='white')
button_tat.place(relx=0.67, rely=0.88, anchor=NW)

# =====================================================

ten=Label(window, text = "nhanb1900295@student.ctu.edu.vn",
          font = ("Times New Roman", 10, "italic"),
          fg='black',
          bg="#BACDDB")
ten.place(relx=0.5, rely=0.998, anchor=S)
window.resizable(False,False)# co dinh kich thuoc mang hinh
window.mainloop()


 