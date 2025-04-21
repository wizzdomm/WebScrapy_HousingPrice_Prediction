import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg
from PIL import ImageTk, Image
from Prediction.divar import prediction
LoginObject = Tk()

def search():
    LoginObject.destroy()
    mainObject = prediction()
    mainObject.PredictionLoad()

LoginObject.geometry('530x530')
right = int(LoginObject.winfo_screenwidth() / 2 - 500/2)
down = int(LoginObject.winfo_screenheight() / 2 - 500/2)
LoginObject.geometry('+{}+{}'.format(right,down))
LoginObject.title('Login Form')
LoginObject.resizable(0,0)
LoginObject.configure(bg='#ffffff')
canvas = Canvas(LoginObject,bg="#ffffff",height=500,width=542,bd=0,highlightthickness=0,relief="ridge")
canvas.place(x=0, y=0)
background_img = ImageTk.PhotoImage(Image.open("Image/divar.jpg"))


background = canvas.create_image(240,190,image=background_img)
btnimage =PhotoImage(file='Image/search.png')
btnEnter = Button(LoginObject,image=btnimage,borderwidth=0,highlightthickness=0,command=search,relief="flat",bg="#ffffff")
btnEnter.place(x=150,y=427,width=200,height=107)



LoginObject.mainloop()