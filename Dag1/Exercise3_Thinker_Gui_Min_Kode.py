from tkinter import *

#Funktioner.
def changeText():
    input = entry.get();
    lbl.configure(text=input)
    entry.delete(0, END)

#Laver et lille vindue
window = Tk()
window.title("Min mega fede awesome app 2000")

#Opretter frame
frame = Frame(window)
frame.pack();

#Tilføjer label og kanpper til Framet.
lbl = Label(frame, text="Ja tjaaak så er vi i gang\n\n")
lbl.pack()

btn = Button(frame, text="LUK", fg="red", command=quit)
btn.pack()

btn2 = Button(frame, text="Ændre tekst", command=changeText)
btn2.pack()

#tilføjer textfield med scrolebar.
S = Scrollbar(frame, orient='vertical')
txt = Text(frame, height=10, width=50, yscrollcommand=S.set)

S.pack(side=RIGHT, fill=Y)
txt.pack(side=LEFT, fill=Y)

#Laver ny et nyt frame.
frame2 = Frame(window)
frame2.pack()

label2 = Label(frame2, text="Indtast navn:")

#vi angiver padx og pady for at få lidt plads
label2.pack(side=LEFT, padx=10, pady=10)

#så skal vi have vores input felt ind
entry = Entry(frame2)
entry.pack(side=LEFT, padx=10, pady=10)




window.mainloop()