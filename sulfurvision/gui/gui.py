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
        self.bar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y", expand=False)
        self.canvas.create_window(0, 0, window=self.frame, anchor="nw")

        self.frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfig(
                self.canvas.find_withtag("all")[0], width=event.width
            ),
        )
        self.canvas.bind("<Enter>", self.bindwheel)
        self.canvas.bind("<Leave>", self.unbindwheel)

    def bindwheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self.scroll)
        self.canvas.bind_all("<Button-4>", self.scroll)
        self.canvas.bind_all("<Button-5>", self.scroll)

    def unbindwheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def scroll(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class AffineTransformFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        affine = kwargs.pop("affine", None)
        super().__init__(*args, **kwargs)

        self.vars = [tk.DoubleVar(value=0) for _ in range(6)]
        self.boxes = [
            tk.Entry(
                self,
                textvariable=self.vars[i],
                validate="all",
                validatecommand=(SulfurGui.validate_float, "%P"),
            )
            for i in range(6)
        ]
        for i, box in enumerate(self.boxes):
            box.grid(row=i // 3, column=i % 3)

        if affine:
            self.set_affine(affine)

    def set_affine(self, affine):
        for var, value in zip(self.vars, affine):
            var.set(value)

    def get(self):
        return np.asarray([var.get for var in self.vars], dtype=np.float64)


class ColorPickerFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        color = np.asarray(kwargs.pop("color", np.zeros(3, np.float64)), dtype=np.int32)
        super().__init__(*args, **kwargs)

        self.preview = tk.Label(
            self, bg=f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
        )
        self.preview.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.red_var = tk.IntVar(value=color[0])
        self.red_var.trace_add("write", self.color_callback)
        self.red = tk.Entry(
            self,
            textvariable=self.red_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.red.grid(row=1, column=0)
        self.green_var = tk.IntVar(value=color[1])
        self.green_var.trace_add("write", self.color_callback)
        self.green = tk.Entry(
            self,
            textvariable=self.green_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.green.grid(row=1, column=1)
        self.blue_var = tk.IntVar(value=color[2])
        self.blue_var.trace_add("write", self.color_callback)
        self.blue = tk.Entry(
            self,
            textvariable=self.blue_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.blue.grid(row=1, column=2)
        self.button = tk.Button(self, text="Pick Color", command=self.pick_color)
        self.button.grid(row=0, column=2)

    def pick_color(self):
        color = colorchooser.askcolor()
        if color[0]:
            self.set_color(color[0])

    def set_color(self, color):
        self.preview.config(bg=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        self.red_var.set(color[0])
        self.green_var.set(color[1])
        self.blue_var.set(color[2])

    def color_callback(self, *_):
        try:
            red = self.red_var.get()
            green = self.green_var.get()
            blue = self.blue_var.get()
            colorstr = f"#{red:02x}{green:02x}{blue:02x}"
            self.preview.config(bg=colorstr)
        except tk.TclError:
            ...


class VariationFrame(tk.Frame):
    """Presents all parameters of one variation in a transform:
    - Weight
    - Params
    """

    def __init__(self, *args, **kwargs):
        variation = kwargs.pop("variation", None)
        transform = kwargs.pop("transform", None)
        super().__init__(*args, **kwargs)
        self.label = tk.Label(
            self, anchor="w", font=("Arial", 18, "bold"), relief="raised"
        )
        self.label.grid(row=0, columnspan=2, pady=8)

        self.weight_label = tk.Label(self, text="Weight:", anchor="w")
        self.weight_label.grid(row=1, column=0, sticky="w")
        self.weight_var = tk.DoubleVar()
        self.weight_box = tk.Entry(
            self,
            textvariable=self.weight_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.weight_box.grid(row=1, column=1, sticky="e")

        self.param_labels = []
        self.param_boxes = []
        self.param_vars = []
        self.params_frame = tk.Frame(self)
        self.params_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        if variation:
            self.load(variation, transform)

    def load(self, variation: variations.Variation, transform: pysulfur.Transform):
        self.label.config(text=variation.name)

        self.param_vars.clear()
        for child in self.params_frame.winfo_children():
            child.destroy()
        for i in range(variation.num_params):
            label = tk.Label(self.params_frame, text=f"Param #{i}")
            label.grid(row=i, column=0, sticky="w")
            self.param_labels.append(label)

            param_var = tk.DoubleVar(
                value=(
                    0 if not transform else transform.params[variation.params_base + i]
                )
            )
            self.param_vars.append(param_var)

            box = tk.Entry(
                self.params_frame,
                textvariable=param_var,
                validate="all",
                validatecommand=(SulfurGui.validate_float, "%P"),
            )
            box.grid(row=i, column=1, sticky="e")
            self.param_boxes.append(box)

        if transform:
            index = variations.Variation.variations_map[variation.name]
            self.weight_var.set(transform.weights[index])


class TransformFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        self.transform = kwargs.pop("transform", None)
        super().__init__(*args, **kwargs)

        self.prob_label = tk.Label(self, text="Probability")
        self.prob_label.grid(row=0, column=0, sticky="w")
        self.prob_var = tk.DoubleVar(value=0)
        self.prob_box = tk.Entry(
            self,
            textvariable=self.prob_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.prob_box.grid(row=0, column=1, sticky="e")

        self.affine_label = tk.Label(self, text="Affine Transform:")
        self.affine_label.grid(row=1, column=0, sticky="w")
        self.affine_frame = AffineTransformFrame(self)
        self.affine_frame.grid(row=2, column=0, columnspan=2, sticky="w")

        # self.color_frame = ColorPickerFrame(self)
        # self.color_frame.grid(row=3, column=0, columnspan=2)
        self.color_label = tk.Label(self, text="Color:")
        self.color_label.grid(row=3, column=0)
        self.color_var = tk.DoubleVar(value=0)
        self.color_box = tk.Entry(
            self,
            textvariable=self.color_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.color_box.grid(row=3, column=1)

        self.speed_label = tk.Label(self, text="Speed:")
        self.speed_label.grid(row=4, column=0, sticky="w")
        self.speed_var = tk.DoubleVar(value=0)
        self.speed_box = tk.Entry(
            self,
            textvariable=self.speed_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.speed_box.grid(row=4, column=1, sticky="e")

        self.var_editor = VariationFrame(self, variation=variations.variation_linear)
        self.var_editor.grid(row=1, column=2, rowspan=4)

        self.dropdown = ttk.Combobox(
            self, values=[v.name for v in variations.Variation.variations]
        )
        self.dropdown.bind(
            "<<ComboboxSelected>>",
            lambda _: self.var_editor.load(
                variations.Variation.variations[
                    variations.Variation.variations_map[self.dropdown.get()]
                ],
                None,
            ),
        )
        self.dropdown.set(variations.Variation.variations[0].name)
        self.dropdown.grid(row=0, column=2)

        if self.transform:
            self.load(self.transform)

    def variation_selected(self, _):
        # Save old values
        self.update_current_variation()

        # Update editor to new variation
        self.var_editor.load(
            variations.Variation.variations[
                variations.Variation.variations_map[self.dropdown.get()]
            ],
            self.transform,
        )

    def update_current_variation(self):
        old_name = self.var_editor.label.cget("text")
        old_index = variations.Variation.variations_map[old_name]
        old_variation = variations.Variation.variations[old_index]
        self.transform.weights[old_index] = self.var_editor.weight_var.get()
        for i in range(old_variation.num_params):
            self.transform.params[old_variation.params_base + i] = (
                self.var_editor.param_vars[i].get()
            )

    def load(self, transform: pysulfur.Transform):
        if not transform:
            return
        if self.transform is transform:
            return
        self.transform = transform
        self.affine_frame.set_affine(transform.affine)
        self.color_var.set(transform.color)
        self.speed_var.set(transform.color_speed)
        self.prob_var.set(transform.probability)
        self.var_editor.load(variations.Variation.variations[0], transform)

    def update(self):
        if not self.transform:
            return
        self.transform.affine = self.affine_frame.get()
        self.transform.color = self.color_var.get()
        self.transform.color_speed = self.speed_var.get()
        self.transform.probability = self.prob_var.get()
        self.update_current_variation()


class KeyframeFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO all of this


def main():
    testtf = pysulfur.Transform(
        variations.Variation.as_weights({"variation_linear": 1}),
        variations.Variation.as_params({}),
        np.array([1, 0, 0, 0, 1, 0], dtype=np.float64),
        1,
        0,
    )

    root = tk.Tk()
    gui = SulfurGui(root)
    gui.pack(expand=True, fill="both")
    scrollable = ScrollableFrame(gui)
    tf = TransformFrame(scrollable.frame)
    tf.load(testtf)
    tf.pack(fill="both", expand=True)
    scrollable.pack(fill="both", expand=True)
    tk.mainloop()


if __name__ == "__main__":
    main()
