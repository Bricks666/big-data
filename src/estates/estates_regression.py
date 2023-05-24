from tkinter import \
    Tk, Entry, Label, \
    Button, Frame, CENTER, \
    font, Variable, StringVar, \
    IntVar, N, OptionMenu, DoubleVar, \
    DISABLED, Radiobutton
from typing import \
    NamedTuple, Dict, Any,  \
    Callable, List, Literal, Union
from pandas import DataFrame
from joblib import load
import requests
from pathlib import Path
from os import getcwd
import json


# model = load('./models/estates_regressor.pkl')
encoders_path = Path(getcwd()) / 'src' / 'estates' / 'estates_encoders.json'


encoders = load(encoders_path)

app = Tk()
app.title('Price counter')

mainframe = Frame(app, padx=10, pady=10)


def textinput(root: Frame, text: str, textvariable: Variable):
    label = Label(root, text=text, anchor=N)
    entry = Entry(root, textvariable=textvariable)

    return (label, entry)


class Option(NamedTuple):
    label: str
    value: Any


def select(root: Frame, text: str, textvariable: Variable, options: List[Option]):
    label = Label(root, text=text, anchor=N)

    variable_labels = list(map(lambda option: option.label, options))

    option_menu = OptionMenu(root, textvariable, *
                             variable_labels)

    return (label, option_menu)


def radiobuttons(root: Frame, text: str, textvariable: Variable, options: List[Option]):
    label = Label(root, text=text, anchor=N)

    buttons_frame = Frame(root)
    buttons = []

    for index, option in enumerate(options):
        button = Radiobutton(
            buttons_frame, variable=textvariable, value=option.value, text=option.label)
        buttons.append(button)
        button.grid(column=index, row=0)

    return (label, buttons_frame, buttons)


VariableCreator = Callable[[Frame, Any], Variable]


class TextFieldDescription(NamedTuple):
    type: Literal['text']
    name: str
    label: str
    variable_creator: VariableCreator
    defaultValue: Any


class SelectDescription(NamedTuple):
    type: Literal['select']
    name: str
    label: str
    variable_creator: VariableCreator
    defaultValue: Any
    options: List[Option]


class RadioDescription(NamedTuple):
    type: Literal['radio']
    name: str
    label: str
    variable_creator: VariableCreator
    defaultValue: Any
    options: List[Option]


FieldDescription = Union[TextFieldDescription,
                         SelectDescription, RadioDescription]
building_ages = encoders['building_age'].classes_

building_age_map = {value: index for index, value in enumerate(building_ages)}
building_age_options = {Option(value, index)
                        for index, value in enumerate(building_ages)}

floor_no_dict = {
    '2': 2, '20 ve üzeri': 20, 'Yüksek Giriş': 1, '10': 10, '14': 14, 'Asma Kat': -1,
    'Bahçe katı': 1, '11': 11, '3': 3, '13': 13, '7': 7, '16': 16, 'Müstakil': 1, 'Zemin Kat': 1,
    '19': 19, '4': 4, '5': 5, 'En Üst Kat': -1, '8': 8, '15': 15, '1': 1, 'Giriş Katı': 1, '9': 9,
    'Çatı Katı': -1, '12': 12, '17': 17, '6': 6, 'Kot 4': 4, 'Kot 2': 2, 'Kot 1': 1, 'Kot 3': 3,
    '18': 18, 'Teras Kat': 1, 'Komple': -1, 'Bodrum Kat': 0,
    '10-20 arası': 10,
}
floor_no_options = [
    Option(value, value) for value in floor_no_dict]
listing_type_labels = {
    1: 'Sell',
    2: 'Rent',
    3: 'Other'
}


listing_type_options: List[Option] = [
    Option(listing_type_labels[number], number) for number in range(1, 4)]

addresses = list(encoders['address'].classes_)
addresses_dict = {value: index for index, value in enumerate(addresses)}
sorted_addresses =sorted(addresses)
addresses_options = {Option(value, value) for _, value in enumerate(sorted_addresses)}

fields: Dict[str, FieldDescription] = {
    'tom': TextFieldDescription(type='text', name='tom', label='Time on market(in days)', variable_creator=lambda root, value: IntVar(root, value), defaultValue=1),
    'building_age': SelectDescription(type='select', name='building_age', label='How old your house', variable_creator=lambda root, value: StringVar(root, value), defaultValue=building_ages[0], options=building_age_options),
    'floor_no': SelectDescription(type='select', name='floor_no', label='Number of floor', variable_creator=lambda root, value:  StringVar(root, value), defaultValue=floor_no_dict['1'], options=floor_no_options),
    'listing_type': RadioDescription(type='radio', name='listing_type', label='Listing type', variable_creator=lambda root, value: IntVar(root, value), defaultValue=1, options=listing_type_options),
    'address': SelectDescription(type='select', name='address', label='Address of the house', variable_creator=lambda root, value: StringVar(root, value), defaultValue=sorted_addresses[0],options=addresses_options),
}

variables: Dict[str, Variable] = {}

main_label = Label(
    mainframe, text='How much your house should coast', anchor=CENTER, font=("Helvetica", 12, font.BOLD), padx=10, pady=10)
main_label.grid()

result_frame = Frame(mainframe)
result_frame.grid()
predict = DoubleVar(result_frame, 0)
predict_label, predict_field = textinput(
    result_frame, 'Predicted coast is:', predict)

predict_label.grid(row=1, column=0)
predict_label.configure(font=("Helvetica", 10, font.NORMAL))
predict_field.grid(row=1, column=1)
predict_field.configure(borderwidth=0, state=DISABLED,
                        font=("Helvetica", 10, font.BOLD))

formframe = Frame(mainframe, padx=10, pady=10, borderwidth=1, relief='solid')


for row, name in enumerate(fields):
    description = fields[name]
    variables[name] = description.variable_creator(
        formframe, description.defaultValue)
    match description.type:
        case 'text':
            label, field = textinput(
                formframe, description.label, variables[name])
        case 'select':
            label, field = select(
                formframe, description.label, variables[name], description.options)
        case 'radio':
            label, field, _ = radiobuttons(
                formframe, description.label, variables[name], description.options)
    label.grid(row=row, column=0)
    field.grid(row=row, column=1)

formframe.grid()

buttonsframe = Frame(mainframe)
buttonsframe.grid()


def prepare_data():
    listing_type = variables['listing_type'].get()
    building_age = building_age_map[variables['building_age'].get()]
    floor_no = floor_no_dict[variables['floor_no'].get()]
    tom = variables['tom'].get()
    address = addresses_dict.get(variables['address'].get())
    return {
        'listing_type': [listing_type],
        'building_age': [building_age],
        'floor_no': [floor_no],
        'tom': [tom],
        'address': [address],
    }


def on_submit():
    data = prepare_data()
    response = requests.post('http://localhost:5000/predict/price',json=data)
    if response.status_code != 200:
        print('Error', response.text)
        return
    print(str(response.content))
    predicted_price, = json.loads(response.text)['predicted']
    predict.set(round(predicted_price, 4))


def on_reset():
    for name, var in variables.items():
        var.set(fields[name].defaultValue)
    predict.set(0)


submit = Button(buttonsframe, text='CALCULATE', command=on_submit)
reset = Button(buttonsframe, text='RESET', command=on_reset)

reset.grid(row=0, column=0)
submit.grid(row=0, column=1)

mainframe.grid()


app.mainloop()
