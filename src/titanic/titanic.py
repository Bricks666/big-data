import tkinter as tk
from typing import TypedDict, Optional, List, Union, Dict, Literal, Any, NamedTuple
from joblib import load
from os import getcwd
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame

filepath = join(getcwd(), 'models', 'titanic.pkl')

print(filepath)

model:RandomForestClassifier = load(filepath)

root = tk.Tk()
root.title('Опросник пассажира титаника')


frame = tk.Frame(root, padx=10, pady=10,borderwidth=5,  relief="ridge")

frame.grid()

answer_frame = tk.Frame(root, padx=10, pady=10)

answer_frame.grid()

class InputPair(TypedDict):
    label:tk.Label
    entry:tk.Entry

def input(root: tk.Frame, label: str = '',
          text_variable: Optional[tk.Variable] = None) -> InputPair :
    Entry = tk.Entry(root, textvariable=text_variable)
    Label = tk.Label(root, text=label)
    return {"label": Label, "entry": Entry}


class Variant(TypedDict):
    label:str
    value: Any

class RadiobuttonPair(TypedDict):
    label:tk.Label
    container: tk.Frame
    radiobuttons:List[tk.Radiobutton]

def radiobutton(root: tk.Frame,variants: List[Variant],
                label: str = '',
                text_variable: Optional[tk.Variable] = None) -> RadiobuttonPair:
    radiobuttons:List[tk.Radiobutton] = []

    Container = tk.Frame(root)
    for index, variant in enumerate(variants):
        radiobutton = tk.Radiobutton(Container, text=variant['label'],
                                           value=variant['value'],
                                           variable=text_variable)
        radiobutton.grid(column=index, row=0)
        radiobuttons.append(radiobutton)
    Label = tk.Label(root, text=label)
    return {"label": Label,"container": Container,  "radiobuttons": radiobuttons}


class FieldDescription(TypedDict):
    type: Literal["text", "number", "radio", "select"]
    variants: Optional[List[Variant]]
    name:str
    label:str
    default_value: Any
    text_variable: tk.Variable

fields:Dict[str, FieldDescription] = {
    "PassendegId":{"type":"text","name": "PassendegId","label":"Id пассажира","text_variable": tk.IntVar(root, 1), "default_value": 1},
    "Pclass":{"type":"radio","name": "Pclass","label":"Класс каюты","text_variable":tk.IntVar(root, 1),"default_value": 1,
     "variants": [{"label": "1", "value": 1},{"label": "2", "value": 2},{"label": "3", "value": 3},]},
    "Name":{"type":"text","name": "Name","label":"Имя","text_variable":tk.StringVar(root, ""),"default_value": ""},
    "Sex":{"type":"radio","name": "Sex","label":"Пол","text_variable":tk.IntVar(root, 0),"default_value": 0,
      "variants": [{"label": "Муж","value": 0}, {"label": "Жен","value": 1}]},
    "Age":{"type":"text","name": "Age","label":"Возраст","text_variable":tk.DoubleVar(root, 18), "default_value": 18},
    "SibSp":{"type":"text","name": "SibSp","label":"SibSp","text_variable":tk.IntVar(root, 0), "default_value": 0},
    "Parch":{"type":"text","name": "Parch","label":"Parch","text_variable":tk.IntVar(root, 0), "default_value": 0},
    "Ticket":{"type":"text","name": "Ticket","label":"Номер билета","text_variable":tk.StringVar(root, ""), "default_value": ""},
    "Fare":{"type":"text","name": "Fare","label":"Стоимость","text_variable":tk.DoubleVar(root, 0), "default_value": 0},
   "Cabin":{"type":"text","name": "Cabin","label":"Номер каюты","text_variable":tk.StringVar(root, ""), "default_value": ""},
    "Embarked":{"type":"radio","name": "Embarked","label":"Код города отправления","text_variable":tk.IntVar(root, 0),"default_value": 0,
     "variants": [{"label": "S", "value":0},{"label": "C", "value": 1},{"label": "Q", "value": 2}]}
}

fields_map: Dict[str, Union[InputPair,RadiobuttonPair]] = {}

for index, value in enumerate(fields.values()):
    match value['type']:
      case 'text':
        fields_map[value['name']] = input(frame, label=value['label'], text_variable=value['text_variable'])
        fields_map[value['name']]['label'].grid(column=0, row=index)
        fields_map[value['name']]['entry'].grid(column=1, row=index)
      case 'radio':
        fields_map[value['name']] = radiobutton(frame, label=value['label'], text_variable=value['text_variable'], variants=value['variants'])
        fields_map[value['name']]['label'].grid(column=0, row=index)
        fields_map[value['name']]['container'].grid(column=1, row=index)


answer = tk.StringVar(root, "")

answer_pair = input(answer_frame,label="Предсказание:", text_variable=answer)
answer_pair['entry'].config(state=tk.DISABLED, border=0)

for index, key in enumerate(answer_pair.keys()):
    answer_pair[key].grid(column=index, row=0)

class ModelAskedFields(NamedTuple):
    Fare: int
    Sex: Any
    Pclass: Any



def prepare_data():
        df  =DataFrame({
        'Fare':[fields['Fare']['text_variable'].get()],
        'Pclass':[fields['Pclass']['text_variable'].get()],
        'Sex': [fields['Sex']['text_variable'].get()]
        })
        return df

def on_submit():
    prepare_fields = prepare_data()
    print(prepare_fields.values)
    result = model.predict(prepare_fields.values)
    print(result)
    if result[0]:
      answer.set("Died")
    else:
      answer.set("Survived")



def on_reset():
    answer.set("")
    for value in fields.values():
        value['text_variable'].set(value['default_value'])



tk.Button(frame, text='Очистить', command=on_reset).grid(column=0, row=len(fields))
tk.Button(frame, text='Предсказать', command=on_submit).grid(column=1, row=len(fields))


root.mainloop()
