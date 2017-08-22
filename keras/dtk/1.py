import tkinter as tk
from tkinter import Frame
from tkinter import filedialog
import os

class Application(Frame):

    def fileBrowse(self):
        self.filename = filedialog.askopenfilename()
        print(self.filename)
    
    def vis(self):
        os.system('./track_vis ' + self.filename + ' -skip ' + self.skip_num)
    
    def B1(self):
        self.button = tk.Button(root,text = "Browse File",command = self.fileBrowse)
        self.button.pack()
    
    def B2(self):
    	self.button = tk.Button(root,text = "Visualize",command = self.vis)
    	self.button.pack()
    	
    def B3(self):
    	self.label = tk.Label(root, text="Enter skip num.")
    	self.label.pack()
    	self.skip_num = str(tk.StringVar())
    	tk.Entry(root, textvariable=self.skip_num).pack()
    	print(self.skip_num)
		
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.filename = None
        self.B1()
        self.B3()
        self.B2()
        self.pack()
        print(self.skip_num)

root = tk.Tk()
root.title("Image Visualize Program")

app = Application(master=root)
app.mainloop()
