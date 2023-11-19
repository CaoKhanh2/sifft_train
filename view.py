# from tkinter import *
# win = Tk()
# win.title('Hello')
# win.geometry('1280x720')

# win.mainloop()


from tkinter import *
import tkinter.filedialog as filedialog
from sift_Classify import sift_model
from PIL import Image, ImageTk


import imutils
import cv2

root=Tk()
root.title("Nhận diện cử chỉ")
root.resizable(height=False,width=False)
root.minsize(height=720,width=720)

def input():
    input_path = filedialog.askopenfilenames()
    input_entry.delete(0, END)
    input_entry.insert(0, input_path)

def show_img(path):
    img = Image.open(path)
    image = ImageTk.PhotoImage(img)
    image_label.configure(image=image)
    image_label.image = image

current_index = 0
result = []

def output():

    filename = []
    filename = input_entry.get().split()
    images = []

    # for i in range(len(filename)):
    #     images.append(cv2.imread(filename[i]))

    # for i in range(len(images)):
    #     images[i] = imutils.resize(images[i], width=400)
    #     images[i] = imutils.resize(images[i], height=400)


    if(len(filename) != 0):
        
        for i in range (len(filename)):
            if(current_index == 0):
                image_Classify = sift_model()
                result.append(image_Classify.main(filename[i]))
                show_img(filename[current_index])
                lbl.configure(text=result[current_index],font=("Arial Bold", 25))
                lbl.pack()
                print()
            elif(0 <= current_index < len(filename)):
                show_img(filename[current_index])
                lbl.configure(text=result[current_index],font=("Arial Bold", 25))
                lbl.pack()
            else:
                print("Index out of range")

    return result

# result = StringVar()
# image_Classify = sift_model()
# result = image_Classify.main(filename)
# print(result)


def show_next_image():
    global current_index
    current_index = current_index + 1
    output()


top_frame = Frame(root)
bottom_frame = Frame(root)
line = Frame(root, height=1, width=400, bg="grey80", relief='groove')
top_frame.pack(side=TOP)
line.pack(pady=10)
bottom_frame.pack(side=BOTTOM)

input_path = Label(top_frame, text="Đường dẫn ảnh:")
input_entry = Entry(top_frame, text="", width=40)

browse1 = Button(top_frame, text="Chọn ảnh", command=input)
input_path.pack(pady=5)
input_entry.pack(pady=5)
browse1.pack(pady=5)

image_label = Label(root)
image_label.pack(pady=10)


lbl = Label(bottom_frame)
lbl.pack(pady=15, fill=X)

button = Button(bottom_frame, text='Nhận diện',command=output)
button.pack(side=RIGHT,pady=20, fill=X)

button = Button(bottom_frame, text='Next',command=show_next_image)
button.pack(side=RIGHT,pady=25, fill=X)


root.mainloop()


