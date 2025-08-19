import sys
import threading
import tkinter as tk
from tkinter import colorchooser, ttk

import numpy as np
from PIL import Image, ImageTk

from sulfurvision import pysulfur, types, util, variations
from sulfurvision.cl import render


def _pre_validate_type(val: str, t: type) -> bool:
    try:
        t(val)
        return True
    except ValueError:
        return val in [".", "", "-"]


def int_from_var(var: tk.IntVar) -> int:
    try:
        return int(var.get())
    except ValueError:
        return 0


def float_from_var(var: tk.DoubleVar) -> float:
    try:
        return float(var.get())
    except ValueError:
        return 0.0


def default_transform():
    return pysulfur.Transform(
        variations.Variation.as_weights({}),
        variations.Variation.as_params({}),
        types.IdentityAffine,
        1,
        0,
    )


def default_frame(n_transforms, n_colors):
    return render.RenderFrame(
        [default_transform() for _ in range(n_transforms)],
        [[0, 0, 0, 1] for _ in range(n_colors)],
        types.IdentityAffine,
        0,
    )


_PREVIEW_SIZE = 200


class SulfurGui(tk.Frame):
    validate_float = None
    validate_int = None

    def __init__(self, *args, **kwargs):
        self.n_transforms = kwargs.pop("n_transforms", 1)
        self.n_colors = kwargs.pop("n_colors", 1)
        super().__init__(*args, **kwargs)
        self.renderer = render.Renderer(_PREVIEW_SIZE, _PREVIEW_SIZE, 1, 50, 1, 3)

        self.anim_frame = tk.Frame(self)
        self.anim_frame.grid(row=0, column=0, sticky="ns")

        SulfurGui.validate_float = self.register(lambda x: _pre_validate_type(x, float))
        SulfurGui.validate_int = self.register(lambda x: _pre_validate_type(x, int))

        self.frames = [default_frame(self.n_transforms, self.n_colors)]

        self.dropdown = ttk.Combobox(self.anim_frame, values=[], state="readonly")
        self.refresh_dropdown()
        self.dropdown.bind("<<ComboboxSelected>>", self.select_keyframe)
        self.dropdown.grid(row=0, column=0)
        self.dropdown.current(0)

        self.insert_before = tk.Button(
            self.anim_frame,
            text="New Frame\nBefore",
            command=lambda: self.insert_frame(self.dropdown.current()),
        )
        self.insert_before.grid(row=0, column=1)

        self.delete = tk.Button(
            self.anim_frame,
            text="Delete Frame",
            command=lambda: self.delete_frame(self.dropdown.current()),
        )
        self.delete.grid(row=1, column=2)

        self.insert_after = tk.Button(
            self.anim_frame,
            text="New Frame\nAfter",
            command=lambda: self.insert_frame(self.dropdown.current() + 1),
        )
        self.insert_after.grid(row=0, column=2)

        self.tf_label = tk.Label(self.anim_frame, text="#Tranforms:")
        self.tf_label.grid(row=1, column=0)
        self.tf_var = tk.IntVar(value=self.n_transforms)
        self.tf_box = tk.Entry(
            self.anim_frame,
            textvariable=self.tf_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.tf_box.grid(row=1, column=1)
        self.update_button = tk.Button(
            self.anim_frame, text="Update", command=self.update_command
        )
        self.update_button.grid(row=2, column=2)

        self.pal_label = tk.Label(self.anim_frame, text="#Colors:")
        self.pal_label.grid(row=2, column=0)
        self.pal_var = tk.IntVar(value=self.n_colors)
        self.pal_box = tk.Entry(
            self.anim_frame,
            textvariable=self.pal_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.pal_box.grid(row=2, column=1)

        self.preview = tk.Label(self.anim_frame, bg="#ffffff")
        self.preview.grid(row=3, column=2, rowspan=7)

        self.width_label = tk.Label(self.anim_frame, text="Width:")
        self.width_label.grid(row=3, column=0)
        self.width_var = tk.IntVar(value=100)
        self.width_box = tk.Entry(
            self.anim_frame,
            textvariable=self.width_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.width_box.grid(row=3, column=1)

        self.height_label = tk.Label(self.anim_frame, text="Height:")
        self.height_label.grid(row=4, column=0)
        self.height_var = tk.IntVar(value=100)
        self.height_box = tk.Entry(
            self.anim_frame,
            textvariable=self.height_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.height_box.grid(row=4, column=1)

        self.bright_label = tk.Label(self.anim_frame, text="Brightness:")
        self.bright_label.grid(row=5, column=0)
        self.bright_var = tk.DoubleVar(value=20)
        self.bright_box = tk.Entry(
            self.anim_frame,
            textvariable=self.bright_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.bright_box.grid(row=5, column=1)

        self.gamma_label = tk.Label(self.anim_frame, text="Gamma:")
        self.gamma_label.grid(row=6, column=0)
        self.gamma_var = tk.DoubleVar(value=20)
        self.gamma_box = tk.Entry(
            self.anim_frame,
            textvariable=self.gamma_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.gamma_box.grid(row=6, column=1)

        self.vib_label = tk.Label(self.anim_frame, text="Vibrancy:")
        self.vib_label.grid(row=7, column=0)
        self.vib_var = tk.DoubleVar(value=20)
        self.vib_box = tk.Entry(
            self.anim_frame,
            textvariable=self.vib_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.vib_box.grid(row=7, column=1)

        self.seed_label = tk.Label(self.anim_frame, text="Seeds:")
        self.seed_label.grid(row=8, column=0)
        self.seed_var = tk.IntVar(value=20)
        self.seed_box = tk.Entry(
            self.anim_frame,
            textvariable=self.seed_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.seed_box.grid(row=8, column=1)

        self.iter_label = tk.Label(self.anim_frame, text="Iterations:")
        self.iter_label.grid(row=9, column=0)
        self.iter_var = tk.IntVar(value=100)
        self.iter_box = tk.Entry(
            self.anim_frame,
            textvariable=self.iter_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.iter_box.grid(row=9, column=1)

        self.skip_label = tk.Label(self.anim_frame, text="Skip:")
        self.skip_label.grid(row=10, column=0)
        self.skip_var = tk.IntVar(value=10)
        self.skip_box = tk.Entry(
            self.anim_frame,
            textvariable=self.skip_var,
            validate="all",
            validatecommand=(SulfurGui.validate_int, "%P"),
        )
        self.skip_box.grid(row=10, column=1)

        # Row 11: Preview
        # Row 12: Render
        # Row 13: Animate

        self.preview_this = tk.Button(
            self.anim_frame, text="Preview This Frame", command=self.render_preview_now
        )
        self.preview_this.grid(row=11, column=0)
        self.preview_then = tk.Button(
            self.anim_frame,
            text="Preview At t=",
            command=lambda: self.render_preview(self.t_var.get()),
        )
        self.preview_then.grid(row=11, column=1)
        self.t_var = tk.DoubleVar(value=0)
        self.preview_t = tk.Entry(
            self.anim_frame,
            textvariable=self.t_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.preview_t.grid(row=12, column=2)

        self.now_button = tk.Button(
            self.anim_frame, text="Render This Frame", command=self.save_now
        )
        self.now_button.grid(row=12, column=0)
        self.then_button = tk.Button(
            self.anim_frame, text="Render At t=", command=self.save(self.t_var.get())
        )
        self.then_button.grid(row=12, column=1)

        self.rate_label = tk.Label(self.anim_frame, text="Frames Per\ndt=1.0:")
        self.rate_label.grid(row=13, column=0)
        self.rate_var = tk.DoubleVar(value=20)
        self.rate_box = tk.Entry(
            self.anim_frame,
            textvariable=self.rate_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.rate_box.grid(row=13, column=1)
        self.animate = tk.Button(
            self.anim_frame, text="Animate", command=self.animate_command
        )
        self.animate.grid(row=13, column=2)

        self.imp = tk.Button(self.anim_frame, text="Import", command=self.imp_command)
        self.imp.grid(row=10, column=2)
        self.exp = tk.Button(self.anim_frame, text="Export", command=self.exp_command)
        self.exp.grid(row=11, column=2)

        self.keyframe = KeyframeFrame(self, frame=self.frames[self.dropdown.current()])
        self.keyframe.grid(row=0, column=1)
        self.update_keyframe()

        self.render_preview_now()

        self.clipboard = None

    def animate_command(self):
        """ TODO:
        - File dialog to choose output directory
        - Read all relevant variables for rendering
        - Iterate through time using framerate
        - At each time step, render the interpolated flame and save to a numbered image file
        """
        ...

    def imp_command(self):
        """ TODO:
        - Present a file dialog to select a file
        - Load contents as JSON
        - Decode and populate GUI
        """
        ...

    def exp_command(self):
        """ TODO:
        - File dialog to select destination path
        - Convert contents to JSON
        - Save to file
        """
        ...

    def select_keyframe(self, _):
        self.keyframe.update()
        self.refresh_dropdown()
        self.update_keyframe()

    def update_keyframe(self):
        self.keyframe.load(self.frames[self.dropdown.current()])

    def update_command(self):
        self.n_transforms = int_from_var(self.tf_var)
        self.n_colors = int_from_var(self.pal_var)
        self.update_params()

    def update_params(self):
        self.keyframe.update()
        for frame in self.frames:
            if self.n_transforms > len(frame.transforms):
                frame.transforms.extend(
                    [
                        default_transform()
                        for _ in range(self.n_transforms - len(frame.transforms))
                    ]
                )
            elif self.n_transforms < len(frame.transforms):
                frame.transforms = frame.transforms[: self.n_transforms]

            if self.n_colors > len(frame.palette):
                frame.palette.extend(
                    [0, 0, 0, 1] for _ in range(self.n_colors - len(frame.palette))
                )
            elif self.n_colors < len(frame.palette):
                frame.palette = frame.palette[: self.n_colors]
        self.update_keyframe()

    def insert_frame(self, idx):
        self.keyframe.update()
        new_frame = default_frame(self.n_transforms, self.n_colors)
        self.frames.insert(idx, new_frame)
        self.refresh_dropdown()
        self.dropdown.current(idx)
        self.update_keyframe()

    def delete_frame(self, idx):
        if len(self.frames) == 1:
            self.frames[0] = default_frame(self.n_transforms, self.n_colors)
        else:
            self.frames.pop(idx)
        self.dropdown.current(0)
        self.refresh_dropdown()
        self.update_keyframe()

    def refresh_dropdown(self):
        self.dropdown.config(
            values=[
                f"Frame #{i}: +{frame.time:.3f}s" for i, frame in enumerate(self.frames)
            ]
        )

    def pairs_for_splines(self):
        return [(frame, frame.time) for frame in self.frames]

    def render_to_image(
        self,
        w: int,
        h: int,
        supersampling: int,
        seeds: int,
        iters: int,
        skip: int,
        brightness: float,
        gamma: float,
        vibrancy: float,
        t: float,
    ) -> Image.Image:
        self.keyframe.update()
        pairs = self.pairs_for_splines()
        frame = util.spline_step(pairs, t)
        self.renderer.update_to_match(
            w, h, supersampling, seeds, self.n_colors, self.n_transforms
        )
        return self.renderer.render(
            frame.camera,
            frame.transforms,
            frame.palette,
            iters,
            skip,
            vibrancy,
            gamma,
            brightness,
        )

    def allow_rendering(self, allow: bool):
        state = 'normal' if allow else 'disabled'
        self.preview_this.config(state=state)
        self.preview_then.config(state=state)
        self.now_button.config(state=state)
        self.then_button.config(state=state)

    def render_preview(self, time):
        def _func():
            self.allow_rendering(False)
            img = self.render_to_image(
                _PREVIEW_SIZE, _PREVIEW_SIZE, 1, 100, 100, 15, 20, 1, 1, time
            )
            photo = ImageTk.PhotoImage(image=img)
            self.preview.config(image=photo)
            self.preview.image = photo
            self.allow_rendering(True)
        threading.Thread(target=_func).start()

    def render_preview_now(self):
        time = sum(map(lambda x: x.time, self.frames[: self.dropdown.current() + 1]))
        self.render_preview(time)

    def save(self, t: float):
        """ TODO:
        - Present a dialog to choose a target destination file path
        - Read all relevant variables for rendering
        - Render to an Image
        - Save to file
        """
        ...

    def save_now(self):
        time = sum(map(lambda x: x.time, self.frames[: self.dropdown.current() + 1]))
        self.save(time)


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
        return np.asarray([float_from_var(var) for var in self.vars], dtype=np.float64)


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
        color = tuple(map(int, color))
        self.preview.config(bg=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        self.red_var.set(color[0])
        self.green_var.set(color[1])
        self.blue_var.set(color[2])

    def get_color(self):
        return (
            int_from_var(self.red_var),
            int_from_var(self.green_var),
            int_from_var(self.blue_var),
            1,
        )

    def color_callback(self, *_):
        try:
            red, green, blue = self.get_color()[:3]
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
        self.affine_frame.grid(row=2, column=0, columnspan=3, sticky="w")

        self.copy = tk.Button(self, text="Copy Transform", command=self.copy_command)
        self.copy.grid(row=0, column=2)
        self.paste = tk.Button(self, text="Paste Transform", command=self.paste_command)
        self.paste.grid(row=1, column=2)

        self.color_label = tk.Label(self, text="Color:")
        self.color_label.grid(row=3, column=0, sticky="w")
        self.color_var = tk.DoubleVar(value=0)
        self.color_box = tk.Entry(
            self,
            textvariable=self.color_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.color_box.grid(row=3, column=1, sticky="e")

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
        self.var_editor.grid(row=5, column=1)

        self.dropdown = ttk.Combobox(
            self,
            values=[v.name for v in variations.Variation.variations],
            state="readonly",
        )
        self.dropdown.bind("<<ComboboxSelected>>", self.variation_selected)
        self.dropdown.set(variations.Variation.variations[0].name)
        self.dropdown.grid(row=5, column=0)

        if self.transform:
            self.load(self.transform)

        self.clipboard = None

    def copy_command(self):
        self.update()
        self.clipboard = self.transform.dump_json()

    def paste_command(self):
        if self.clipboard is None:
            return
        transform = pysulfur.Transform.read_json(self.clipboard)
        self.transform.affine = transform.affine
        self.transform.color = transform.color
        self.transform.color_speed = transform.color_speed
        self.transform.probability = transform.probability
        self.transform.weights = transform.weights
        self.transform.params = transform.params
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
        self.transform.weights[old_index] = float_from_var(self.var_editor.weight_var)
        for i in range(old_variation.num_params):
            self.transform.params[old_variation.params_base + i] = (
                self.var_editor.param_vars[i].get()
            )

    def load(self, transform: pysulfur.Transform):
        if not transform:
            return
        self.transform = transform
        self.affine_frame.set_affine(transform.affine)
        self.color_var.set(transform.color)
        self.speed_var.set(transform.color_speed)
        self.prob_var.set(transform.probability)
        self.dropdown.set(variations.Variation.variations[0].name)
        self.var_editor.load(variations.Variation.variations[0], transform)

    def update(self):
        if not self.transform:
            return
        self.transform.affine = self.affine_frame.get()
        self.transform.color = float_from_var(self.color_var)
        self.transform.color_speed = float_from_var(self.speed_var)
        self.transform.probability = float_from_var(self.prob_var)
        self.update_current_variation()
        if abs(self.transform.weights.sum()) > 1e-9:
            self.transform.weights /= self.transform.weights.sum()


class KeyframeFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        self.frame = kwargs.pop("frame")
        super().__init__(*args, **kwargs)
        self.n_colors = len(self.frame.palette)
        self.n_transforms = len(self.frame.transforms)
        self.tf_num = 0
        self.time_label = tk.Label(self, text="Time:")
        self.time_label.grid(row=0, column=0, sticky="w")
        self.time_var = tk.DoubleVar(value=self.frame.time)
        self.time_box = tk.Entry(
            self,
            textvariable=self.time_var,
            validate="all",
            validatecommand=(SulfurGui.validate_float, "%P"),
        )
        self.time_box.grid(row=0, column=1, sticky="e")

        self.camera_label = tk.Label(self, text="Camera:")
        self.camera_label.grid(row=1, column=0, sticky="w")
        self.camera_frame = AffineTransformFrame(self)
        self.camera_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.camera_frame.set_affine(self.frame.camera)

        self.palette_frame = ScrollableFrame(self)
        self.palette_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.color_pickers = []
        for i, color in enumerate(self.frame.palette):
            color_frame = ColorPickerFrame(self.palette_frame.frame, color=color)
            color_frame.grid(row=i, column=0, sticky="ew")
            self.color_pickers.append(color_frame)

        self.copy_frame = tk.Button(self, text="Copy Frame", command=self.copy_command)
        self.copy_frame.grid(row=4, column=0)
        self.paste_frame = tk.Button(
            self, text="Paste Frame", command=self.paste_command
        )
        self.paste_frame.grid(row=4, column=1)

        self.dropdown = ttk.Combobox(
            self,
            values=[f"Transform #{i}" for i in range(self.n_transforms)],
            state="readonly",
        )
        self.dropdown.set(f"Transform #{self.tf_num}")
        self.dropdown.grid(row=5, column=0, sticky="w")
        self.dropdown.bind("<<ComboboxSelected>>", self.transform_selected)
        self.tf_frame = TransformFrame(
            self, transform=self.frame.transforms[self.tf_num]
        )
        self.tf_frame.grid(row=6, column=0, columnspan=2)

        self.clipboard = None

    def copy_command(self):
        self.update()
        self.clipboard = self.frame.dump_json()

    def paste_command(self):
        if self.clipboard is None:
            return
        frame = render.RenderFrame.read_json(self.clipboard)
        if self.n_colors != len(frame.palette) or self.n_transforms != len(frame.transforms):
            print('Color or transform sizes of frames do not match', file=sys.stderr)
            return
        self.frame.__dict__.update(frame.__dict__)
        self.load(self.frame)

    def transform_selected(self, _):
        self.update_current_transform()
        self.tf_num = self.dropdown.current()
        self.tf_frame.load(self.frame.transforms[self.tf_num])

    def update_current_transform(self):
        self.tf_frame.update()

    def update(self):
        self.tf_frame.update()
        self.frame.time = float_from_var(self.time_var)
        self.frame.camera = self.camera_frame.get()
        for i in range(self.n_colors):
            color_frame = self.color_pickers[i]
            self.frame.palette[i] = color_frame.get_color()
        total_prob = sum(map(lambda tf: tf.probability, self.frame.transforms))
        if abs(total_prob) > 1e-9:
            for tf in self.frame.transforms:
                tf.probability /= total_prob
        self.tf_frame.load(self.frame.transforms[self.dropdown.current()])

    def load(self, frame):
        self.frame = frame
        self.n_colors = len(self.frame.palette)
        self.n_transforms = len(self.frame.transforms)
        if self.tf_num >= self.n_transforms:
            self.tf_num = self.n_transforms - 1
        self.time_var.set(self.frame.time)
        self.camera_frame.set_affine(self.frame.camera)
        for i, color in enumerate(self.frame.palette):
            if i >= len(self.color_pickers):
                self.color_pickers.append(ColorPickerFrame(self.palette_frame.frame))
            color_picker = self.color_pickers[i]
            color_picker.set_color(color)
            color_picker.grid(row=i, column=0)
        for i in range(self.n_colors, len(self.color_pickers)):
            self.color_pickers[i].grid_forget()
        self.dropdown.config(
            values=[f"Transform #{i}" for i in range(self.n_transforms)]
        )
        self.dropdown.set(f"Transform #{self.tf_num}")
        self.tf_frame.load(self.frame.transforms[self.tf_num])


def main():
    root = tk.Tk()
    gui = SulfurGui(root)
    gui.pack(fill="both", expand=True)
    tk.mainloop()


if __name__ == "__main__":
    main()
