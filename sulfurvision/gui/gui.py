import tkinter as tk
from tkinter import colorchooser, ttk

import numpy as np

from sulfurvision import pysulfur, variations

def _pre_validate_type(val: str, t: type) -> bool:
    try:
        t(val)
        return True
    except ValueError:
        return False

class SulfurGui(tk.Frame):
    validate_float = None
    validate_int = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SulfurGui.validate_float = self.register(lambda x: _pre_validate_type(x, float))
        SulfurGui.validate_int = self.register(lambda x: _pre_validate_type(x, int))

class ScrollableFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.canvas = tk.Canvas(self)
        self.frame = tk.Frame(self.canvas)
        self.bar = tk.Scrollbar(self, orient = 'vertical', command = self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)

        self.canvas.pack(side='left', fill='both', expand=True)
        self.bar.pack(side='right', fill='y', expand=False)
        self.canvas.create_window(0, 0, window=self.frame, anchor='nw')

        self.frame.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfig(self.canvas.find_withtag('all')[0], width=event.width))

class AffineTransformFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        affine = kwargs.pop('affine', None)
        super().__init__(*args, **kwargs)

        self.vars = [tk.DoubleVar(value=0) for _ in range(6)]
        self.boxes = [tk.Entry(self, textvariable=self.vars[i], validate='all', validatecommand=(SulfurGui.validate_float, '%P')) for i in range(6)]
        for i, box in enumerate(self.boxes):
            box.grid(row=i//3, column=i%3)
        
        if affine:
            self.set_affine(affine)
    
    def set_affine(self, affine):
        for var, value in zip(self.vars, affine):
            var.set(value)

class ColorPickerFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        color = np.asarray(kwargs.pop('color', np.zeros(3, np.float32)), dtype=np.int32)
        super().__init__(*args, **kwargs)

        self.preview = tk.Label(self, bg=f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}')
        self.preview.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.red_var = tk.IntVar(value=color[0])
        self.red_var.trace_add('write', self.color_callback)
        self.red = tk.Entry(self, textvariable=self.red_var, validate='all', validatecommand=(SulfurGui.validate_int, '%P'))
        self.red.grid(row=1, column=0)
        self.green_var = tk.IntVar(value=color[1])
        self.green_var.trace_add('write', self.color_callback)
        self.green = tk.Entry(self, textvariable=self.green_var, validate='all', validatecommand=(SulfurGui.validate_int, '%P'))
        self.green.grid(row=1, column=1)
        self.blue_var = tk.IntVar(value=color[2])
        self.blue_var.trace_add('write', self.color_callback)
        self.blue = tk.Entry(self, textvariable=self.blue_var, validate='all', validatecommand=(SulfurGui.validate_int, '%P'))
        self.blue.grid(row=1, column=2)
        self.button = tk.Button(self, text='Pick Color', command=self.pick_color)
        self.button.grid(row=0, column=2)
    
    def pick_color(self):
        color = colorchooser.askcolor()
        if color[0]:
            self.preview.config(bg=color[1])
            self.red_var.set(color[0][0])
            self.green_var.set(color[0][1])
            self.blue_var.set(color[0][2])
    
    def color_callback(self, *_):
        try:
            red = self.red_var.get()
            green = self.green_var.get()
            blue = self.blue_var.get()
            colorstr = f'#{red:02x}{green:02x}{blue:02x}'
            self.preview.config(bg=colorstr)
        except tk.TclError:
            ...

class VariationFrame(tk.Frame):
    """Presents all parameters of one variation in a transform:
    - Weight
    - Params
    - Affine
    - Color
    - Color speed
    """
    def __init__(self, *args, **kwargs):
        variation = kwargs.pop('variation', None)
        transform = kwargs.pop('transform', None)
        super().__init__(*args, **kwargs)
        self.label = tk.Label(self, anchor='w', font=('Arial', 18, 'bold'), relief='raised')
        self.label.grid(row=0, columnspan=2, pady=8)

        self.weight_label = tk.Label(self, text='Weight:', anchor='w')
        self.weight_label.grid(row=1, column=0, sticky='w')
        self.weight_var = tk.DoubleVar()
        self.weight_box = tk.Entry(self, textvariable=self.weight_var, validate='all', validatecommand=(SulfurGui.validate_float, '%P'))
        self.weight_box.grid(row=1, column=1, sticky='e')

        self.param_labels = []
        self.param_boxes = []
        self.param_vars = []
        self.params_frame = tk.Frame(self)
        self.params_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        
        self.affine_label = tk.Label(self, text='Affine Transform:')
        self.affine_label.grid(row=3, column=0, sticky='w')
        self.affine_frame = AffineTransformFrame(self)
        self.affine_frame.grid(row=4, column=0, columnspan=2, sticky='w')

        self.color_frame = ColorPickerFrame(self)
        self.color_frame.grid(row=5, column=0, columnspan=2)

        self.speed_label = tk.Label(self, text='Speed:')
        self.speed_label.grid(row=6, column=0, sticky='w')
        self.speed_var = tk.DoubleVar(value=0)
        self.speed_box = tk.Entry(self, textvariable=self.speed_var, validate='all', validatecommand=(SulfurGui.validate_float, '%P'))
        self.speed_box.grid(row=6, column=1, sticky='e')

        if variation:
            self.load(variation, transform)
    
    def load(self, variation: variations.Variation, transform: pysulfur.Transform):
        self.label.config(text=variation.name)

        self.param_vars.clear()
        for child in self.params_frame.winfo_children():
            child.destroy()
        for i in range(variation.num_params):
            label = tk.Label(self.params_frame, text=f'Param #{i}')
            label.grid(row=i, column=0, sticky='w')
            self.param_labels.append(label)

            param_var = tk.DoubleVar(value = 0 if not transform else transform.params[variation.params_base + i])
            self.param_vars.append(param_var)

            box = tk.Entry(self.params_frame, textvariable=param_var, validate='all', validatecommand=(SulfurGui.validate_float, '%P'))
            box.grid(row=i, column=1, sticky='e')
            self.param_boxes.append(box)
        

def main():
    root = tk.Tk()
    gui = SulfurGui(root)
    gui.pack(expand=True, fill='both')
    scrollable = ScrollableFrame(gui)
    for variation in variations.Variation.variations:
        vf = VariationFrame(scrollable.frame)
        vf.load(variation, None)
        vf.pack(fill='both', expand=True)
    scrollable.pack(fill='both', expand=True)
    tk.mainloop()

if __name__ == '__main__':
    main()
