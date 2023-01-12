import PySimpleGUI as sg
import compare
myfont=('Roboto Mono',13)
output=sg.Text(font=myfont)

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Enter the nepali song that you want to find genre of',font=myfont)],
            [sg.Text('song name',font=myfont), sg.InputText(), sg.FileBrowse()],
            [sg.Button('Ok'), sg.Button('Cancel')],
            [output]]

# Create the Window
window = sg.Window('music genre classification system', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    genre=compare.classifier(values[0])
    output.update(genre)

window.close()