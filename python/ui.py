import PIL.Image
import PIL.ImageDraw 
from PIL import ImageTk, Image
import sys

from Tkinter import *
import tkFileDialog as FD
import ttk
import tkMessageBox


from darknet import *

import cv2


net = load_net(b"../cfg/yolov3.cfg", b"../yolov3.weights", 0)
meta = load_meta(b"../cfg/coco.data")


def startGUI():
    global gbRoot, gbTop
    gbRoot = Tk()
    gbTop = New_Toplevel (gbRoot)
    gbRoot.resizable(width=False, height=False)
    gbRoot.mainloop()

class New_Toplevel:
    def __init__(self, gbTop=None):
        # Tao form giao dien chinh
        gbTop.geometry("600x600+400+50")
        gbTop.title("You Only Look Once - YOLO - Duong Dang Khoa - 151251")
        gbTop.configure(background="#d9d9d9")
        gbTop.configure(highlightbackground="#d9d9d9")
        gbTop.configure(highlightcolor="black")

        # Tao khung dung load anh
        self._frameLoadImage = Frame(gbTop)
        self._frameLoadImage.place(relx=0.03, rely=0.03, height=460, width=460)
        self._frameLoadImage.configure(relief=RIDGE)
        self._frameLoadImage.configure(borderwidth="4")
        self._frameLoadImage.configure(background="#ffffff")

        # Tao event click open image
        self._pathFileOpen = ""    
        def clickOpenEvent():
            self._pathFileOpen = FD.askopenfilename(title='open')
            self._imageOpened = PIL.Image.open(self._pathFileOpen)
            self._imageOpened = self._imageOpened.resize((460, 460), PIL.Image.ANTIALIAS)
            self._imageOpened = ImageTk.PhotoImage(self._imageOpened)
    
            for widget in self._frameLoadImage.winfo_children():
                widget.destroy()

            self._labelImage = Label(self._frameLoadImage, image=self._imageOpened)
            self._labelImage.image = self._imageOpened
            self._labelImage.pack()

                
        self._buttonOpen = Button(gbTop, command=clickOpenEvent)
        self._buttonOpen.place(relx=0.82, rely=0.03, height=53, width=96)
        self._buttonOpen.configure(text='''Open Image''')

        self._listboxShowResult = Listbox(gbTop)
        self._listboxShowResult.place(relx=0.03, rely=0.82, relheight=0.15, relwidth=0.94)
        self._listboxShowResult.configure(relief=RIDGE)
        self._listboxShowResult.configure(borderwidth="4")
        self._listboxShowResult.configure(background="#ffffff")

        
        
        self._pathSavedCache = b"../data/cache/predict.jpg"
        
        def clickRunEvent():
            
            self._listboxShowResult.delete(0,END)
            self._listboxShowResult.insert(END,"Processing... Please wait for a few minutes")

            result = ["RESULT:"]
            predict = detect(net, meta, self._pathFileOpen) 
            self._imageOpened = cv2.imread(self._pathFileOpen)
            self._imageOverlay = self._imageOpened.copy()
            color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255)]
            color = 0

            for i in range(len(predict)):
                result.append(str(predict[i][0].decode()) + ": " + str(predict[i][1]*100) +  "%; ")
                x = int(predict[i][2][0])
                y = int(predict[i][2][1])
                w = int(predict[i][2][2])
                h = int(predict[i][2][3])
                
                cv2.rectangle(self._imageOverlay, (x-w/2,y-h/2), (x+w/2,y+h/2), color_list[color], 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self._imageOverlay,predict[i][0], (x-w/2,y-h/2), font, 1, color_list[color], 2, cv2.LINE_AA)
                self.opacity = 1.0
                cv2.addWeighted(self._imageOverlay, self.opacity, self._imageOpened, 1 - self.opacity, 0, self._imageOpened)
                cv2.imwrite(self._pathSavedCache, self._imageOpened)
                color+=1
                if(color >= len(color_list)):
                    color = 0

            self._imagePredicted = PIL.Image.open(self._pathSavedCache)
            self._imagePredicted = self._imagePredicted.resize((460, 460), PIL.Image.ANTIALIAS)
            self._imagePredicted = ImageTk.PhotoImage(self._imagePredicted)
            
            for widget in self._frameLoadImage.winfo_children():
                widget.destroy()

            self._labelImage = Label(self._frameLoadImage, image=self._imagePredicted)
            self._labelImage.image = self._imagePredicted
            self._labelImage.pack()
            self._listboxShowResult.delete(0,END)
            for i in result:
                self._listboxShowResult.insert(END,i)
                        
            

        self._buttonRun = Button(gbTop, command=clickRunEvent)
        self._buttonRun.place(relx=0.82, rely=0.17, height=53, width=96)
        self._buttonRun.configure(text='''Run''')

        def clickSaveEvent():
            fileName = FD.asksaveasfilename(title=u'Save file', filetypes=[("JPG", ".jpg")])
            imageSaved = PIL.Image.open(self._pathSavedCache)
            imageSaved.save(str(fileName))


        self._buttonSave = Button(gbTop, command=clickSaveEvent)
        self._buttonSave.place(relx=0.82, rely=0.31, height=53, width=96)
        self._buttonSave.configure(text='''Save as''')

        def clickAboutEvent():
            tkMessageBox.showinfo("Final Semester Project - YOLO", "Full Name: Duong Dang Khoa \
                                                                    ID: 1512251")

        self._buttonAbout = Button(gbTop, command=clickAboutEvent)
        self._buttonAbout.place(relx=0.82, rely=0.45, height=53, width=96)
        self._buttonAbout.configure(text='''About''')

        

if __name__ == '__main__':
    startGUI()