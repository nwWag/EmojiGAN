from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# Ein Fenster erstellen
fenster = Tk()
# Den Fenstertitle erstellen
fenster.title("Emoji GAN")
# Initial Window Size and Position
fenster.geometry("550x900") #Width x Height
# Background Color
fenster.configure(bg='white')

# Functions
def fileDialog():
    '''
    File Dialog for browsing a picture
    Also places the picture to the given preview window
    '''
    file = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype = (("png files",".png"), ("jpeg files","*.jpg")) )

    image = Image.open(file)
    image = image.resize((300, 200))
    tkimage = ImageTk.PhotoImage(image)
    preview=Label(fenster, image = tkimage)
    preview.image = tkimage
    preview.pack()
    preview.place(x = 200, y = 30)

def fileDialogStyle():
    '''
    File Dialog for browsing a style
    Also places the style to the given preview window
    '''
    file = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype = (("png files",".png"), ("jpeg files","*.jpg")) )

    image = Image.open(file)
    image = image.resize((150, 150))
    tkimage = ImageTk.PhotoImage(image)
    preview=Label(fenster, image = tkimage)
    preview.image = tkimage
    preview.pack()
    preview.place(x = 275, y = 300)


def fileDialogParams():
    '''
    File Dialog for browsing a parameter file
    '''
    file = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype = (("all files","*.*")))

    # code here
    
def generateEmoji():
    '''
    Function to generate an emoji from fileDialog(), fileDialogStyle() and fileDialogParams()
    '''
    return 0

# Button Pictures
'''
Pictures located in folder gui_data
'''
upload = Image.open(r"gui_data\Button_ImportPicture.png")
upload = upload.resize((150, 50))
upload = ImageTk.PhotoImage(upload)
generate = Image.open(r"gui_data\Button_GenerateEmoji.png")
generate = generate.resize((150, 50))
generate = ImageTk.PhotoImage(generate)
background = Image.open(r"gui_data\data\background.png")
background = background.resize((300, 200))
background = ImageTk.PhotoImage(background)
style = Image.open(r"gui_data\data\Button_ImportStyle.png")
style = style.resize((150, 50))
style = ImageTk.PhotoImage(style)
params = Image.open(r"gui_data\data\Button_ImportParams.png")
params = params.resize((150, 50))
params = ImageTk.PhotoImage(params)

# create Buttons
'''
Buttons 
'''
browse_button = Button(fenster, text="Import Picture", command=fileDialog, image = upload)
generate_button = Button(fenster, text="Generate Emoji", command=generateEmoji(), image = generate)
style_button = Button(fenster, text="Import Style", command=fileDialogStyle, image = style)
params_button = Button(fenster, text="Import Style", command=fileDialogParams, image = params)

# create Labels
'''
Labels 
'''
background_prev = Label(fenster,image=background)
background_emojigen = Label(fenster,image=background)
background_style_prev = Label(fenster,image=background)
background_params_prev = Label(fenster,image=background)


# components
'''
Put everything together and insert components at the right position  
'''
browse_button.place(x = 20, y = 100, width=150, height=50)
background_prev.place(x = 200, y = 30)
generate_button.place(x = 20, y = 690, width=150, height=50)
background_emojigen.place(x = 200, y = 620)
style_button.place(x = 20, y = 350,  width=150, height=50)
background_style_prev.place(x = 275, y = 300, width=150, height=150)
params_button.place(x = 20, y = 500, width=150, height=50)

# start main loop
fenster.mainloop()