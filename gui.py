#from tkinter import *
import tkinter

# creates the main window object, defines its name, and default size
main = tkinter.Tk()
main.title('Authentication Box')
main.geometry('225x150')

def clear_widget(event):

    # will clear out any entry boxes defined below when the user shifts
    # focus to the widgets defined below
    if username_box == main.focus_get() and username_box.get() == 'Enter Username':
        username_box.delete(0, tkinter.END)
    elif password_box == password_box.focus_get() and password_box.get() == '     ':
        password_box.delete(0, tkinter.END)

def repopulate_defaults(event):

    # will repopulate the default text previously inside the entry boxes defined below if
    # the user does not put anything in while focused and changes focus to another widget
    if username_box != main.focus_get() and username_box.get() == '':
        username_box.insert(0, 'Enter Username')
    elif password_box != main.focus_get() and password_box.get() == '':
        password_box.insert(0, '     ')

def login(*event):

    # Able to be called from a key binding or a button click because of the '*event'
    print (('Username: ') + username_box.get())
    print (('Password: ') + password_box.get())
    main.destroy()
    # If I wanted I could also pass the username and password I got above to another 
    # function from here.


# defines a grid 50 x 50 cells in the main window
rows = 0
while rows < 10:
    main.rowconfigure(rows, weight=1)
    main.columnconfigure(rows, weight=1)
    rows += 1


# adds username entry widget and defines its properties
username_box = tkinter.Entry(main)
username_box.insert(0, 'Enter Username')
username_box.bind("<FocusIn>", clear_widget)
username_box.bind('<FocusOut>', repopulate_defaults)
username_box.grid(row=1, column=5, sticky='NS')


# adds password entry widget and defines its properties
password_box = tkinter.Entry(main, show='*')
password_box.insert(0, '     ')
password_box.bind("<FocusIn>", clear_widget)
password_box.bind('<FocusOut>', repopulate_defaults)
password_box.bind('<Return>', login)
password_box.grid(row=2, column=5, sticky='NS')


# adds login button and defines its properties
login_btn = tkinter.Button(main, text='Login', command=login)
login_btn.bind('<Return>', login)
login_btn.grid(row=5, column=5, sticky='NESW')


main.mainloop()