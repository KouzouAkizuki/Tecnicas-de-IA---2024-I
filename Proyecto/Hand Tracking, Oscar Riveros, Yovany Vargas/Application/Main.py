import tkinter as tk
from tkinter import Frame, Label
from PIL import Image, ImageTk
from Functions import *

pathNet = os.path.join(os.path.dirname(__file__), 'redEntrenada.pkl')
pathUnal = os.path.join(os.path.dirname(__file__), 'Interface/unal.png')
pathError = os.path.join(os.path.dirname(__file__), 'Interface/error.png')
pathDefault = os.path.join(os.path.dirname(__file__), 'Interface/default.png')
red = joblib.load(pathNet)
width = int(720 * 0.7)

def showHeadline(screen):
    headlineFrame = Frame(screen, bg = "#94b43b")
    headlineFrame.grid(row = 0, column = 0, columnspan = 2, sticky = 'ew')
    unalShield = Image.open(pathUnal)
    unalShield = unalShield.resize((226, 100), Image.LANCZOS)
    unalShield = ImageTk.PhotoImage(unalShield)
    shieldFrame = Frame(headlineFrame)
    shieldFrame.pack(side = 'left', padx = 0, pady = 0)
    shieldLabel = Label(shieldFrame, image = unalShield)
    shieldLabel.config(bg = '#94b43b')
    shieldLabel.image = unalShield
    shieldLabel.pack()
    textFrame = Frame(headlineFrame, bg = "#94b43b")
    textFrame.pack(side = 'left', expand = True, fill = 'both')
    titleLabel = Label(textFrame, text = 'Proyecto Final TIA - Hand Tracking', bg = "#94b43b", font = ("Arial", 20, "bold italic"))
    titleLabel.pack(anchor = 'center', pady = 0)
    name1Label = Label(textFrame, text = 'Oscar Leonardo Riveros Perez', bg = "#94b43b", font = ("Arial", 18, "italic"))
    name1Label.pack(anchor = 'center', pady = 0)
    name2Label = Label(textFrame, text = 'Yovany Esneider Vargas Gutierrez', bg = "#94b43b", font = ("Arial", 18, "italic"))
    name2Label.pack(anchor = 'center', pady = 0)

def captureImage():
    ret, frame = cap.read()
    if ret:
        whichGesture(frame)

def whichGesture(frame):
    case = handTracking(red, frame)
    if case == 0:
        outPutImg = Image.open(pathError)
    else:
        pathCase = os.path.join(os.path.dirname(__file__), 'Interface/' + str(case) + '.png')
        outPutImg = Image.open(pathCase)

    outPutImg = outPutImg.resize((int(1280/3), width))
    imageTk = ImageTk.PhotoImage(outPutImg)
    imagePanel.configure(image = imageTk)
    imagePanel.image = imageTk

def showFrame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, ((1280*2)//3, width))
        img = Image.fromarray(frame)
        imageTk = ImageTk.PhotoImage(image = img)
        cameraPanel.imageTk = imageTk
        cameraPanel.configure(image = imageTk)
    cameraPanel.after(10, showFrame)

screen = tk.Tk()
screen.title('Interfaz, proyecto final')
screen.state('zoomed')
screen.configure(bg = 'white')
containerFrame = tk.Frame(screen)
containerFrame.config(bg = 'white')
containerFrame.grid(row = 1, column = 0, columnspan = 2, sticky = 'nsew', padx = 0, pady = 0)
cameraPanel = tk.Label(containerFrame)
cameraPanel.config(bg = 'white')
cameraPanel.grid(row = 0, column = 0, sticky = 'nsew')
imagePanel = tk.Label(containerFrame)
imagePanel.config(bg = 'white')
imagePanel.grid(row = 0, column = 1, sticky = 'nsew')
captureButton = tk.Button(screen, text = "Clasificar gesto", command = captureImage)
captureButton.config(bg = 'white', font = 8)
captureButton.grid(row = 2, column = 0, columnspan = 2, sticky = 'ew')
screen.grid_rowconfigure(1, weight = 1)
screen.grid_columnconfigure(0, weight = 2)
screen.grid_columnconfigure(1, weight = 1)
defaultImage = Image.open(pathDefault)
defaultImage = defaultImage.resize((int(1280/3), width))
imagenTk = ImageTk.PhotoImage(defaultImage)
imagePanel.configure(image = imagenTk)
imagePanel.image = imagenTk

cap = cv2.VideoCapture(1) # 0 para webcam, 1 para celular

# setDefaultImage()
showHeadline(screen)
showFrame()

screen.mainloop()