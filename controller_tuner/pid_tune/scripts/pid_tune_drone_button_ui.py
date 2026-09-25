#!/usr/bin/env python3

import math
import signal
import threading
import time
from collections import deque

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkfont
from tkinter import messagebox
import rclpy
from rclpy.executors import ExternalShutdownException, ShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from controller_msg.msg import PIDTune
from error_msg.msg import Error
import yaml
import os
from ament_index_python.packages import get_package_share_directory

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BG = '#000000'
CARD_BG = '#111318'
TEXT_LIGHT = '#f5f5f5'
MUTED = '#7d8590'
ACCENT = '#2f6fed'


def _darken(hex_color, factor=0.82):
    """Return a slightly darker shade of hex_color, for button press feedback."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}'


class PIDTuningApp(Node):
    MULT_DEFAULTS = {'Kp': 0.03, 'Ki': 0.008, 'Kd': 0.6}
    MAX_WINDOW_SECONDS = 25
    GRAPH_UPDATE_MS = 100
    MIN_GRAPH_SIZE = (560, 240)  # floor for the plot area, enforced through the window's minsize
    # Y autoscale: the axis follows the amplitude currently on screen.
    Y_MIN_SPAN = 0.02       # tightest zoom, in error units: below this the axis stops shrinking
    Y_PAD = 0.15            # fraction of the visible range kept as headroom above and below
    Y_TICK_TARGET = 4       # aim for roughly this many grid steps across the axis
    Y_SHRINK_RATIO = 1.5    # zoom in once the axis is this much wider than the trace needs
    Y_SHRINK_HOLD_S = 0.5   # ...and that calmer amplitude has held for this long
    # series key -> (buffer attribute, line color, display label)
    SERIES = {
        'throttle': ('throttle_buf', '#ff5c5c', 'Throttle'),
        'pitch': ('pitch_buf', '#4cd964', 'Pitch'),
        'roll': ('roll_buf', '#5ac8fa', 'Roll'),
    }

    def __init__(self, root):
        super().__init__('drone_pid_tuner')
        self.throttle_pub = self.create_publisher(PIDTune, "/throttle_pid", 10)
        self.pitch_pub = self.create_publisher(PIDTune, "/pitch_pid", 10)
        self.roll_pub = self.create_publisher(PIDTune, "/roll_pid", 10)
        self.pos_error_sub = self.create_subscription(Error, "/pos_error", self._pos_error_callback, 10)

        self.root = root
        self.root.title('PID Tuning - Quadcopter')
        self.root.attributes("-topmost", True)
        self.root.resizable(True, True)
        self.root.configure(bg=BG)

        self._create_fonts()

        self.entries = {}
        self.step_entries = {}
        self.multiplier_entries = {}
        self.computed_labels = {}
        # Keep the entries' StringVars alive: nothing else references them once
        # _create_pid_row/_create_section return, so without this they get
        # garbage-collected and silently drop their write traces.
        self._value_vars = {}
        self._mult_vars = {}
        self._saved_values = self._load_saved_values()

        # Rolling buffers for the /pos_error graph (trimmed to MAX_WINDOW_SECONDS in the callback).
        # Written by the executor thread, read by the Tk thread, so they need the lock.
        self.start_time = time.time()
        self.paused = False
        self._data_lock = threading.Lock()
        self.time_buf = deque()
        self.throttle_buf = deque()
        self.pitch_buf = deque()
        self.roll_buf = deque()

        self._load_icons()
        self._create_widgets()
        self._apply_responsive_geometry()

        # Spin ROS on its own thread: it blocks on the wait set instead of being polled from the
        # Tk timer, so every /pos_error message is consumed promptly and idle costs no CPU.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()

        # rclpy.init() installs its own SIGINT handler, which shuts the ROS context down but
        # never sets Python's interrupt flag -- so Tk's mainloop never learns about the Ctrl-C
        # and the window stays up long after the node is dead. Take the signal back and route
        # it through the same teardown the window's close button uses.
        self._closing = False
        self._interrupted = False
        signal.signal(signal.SIGINT, self._on_sigint)

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._update_graph()
        self.root.after(100, self._publish_pid_values)

    def _publish_pid_values(self):
        if self._closing:
            return

        self.throttle_pub.publish(self.pid_values("throttle"))
        self.pitch_pub.publish(self.pid_values("pitch"))
        self.roll_pub.publish(self.pid_values("roll"))

        self.root.after(100, self._publish_pid_values)

    def _spin_ros(self):
        """Run the ROS executor until shutdown (background thread)."""
        try:
            self._executor.spin()
        except (ExternalShutdownException, ShutdownException):
            pass

    def _on_sigint(self, _signum, _frame):
        """Ctrl-C: flag the interrupt; the graph tick picks it up on the Tk thread.

        Tk owns the main thread inside mainloop(), so tearing the window down from here
        would run Tk calls from a signal handler. The 100 ms tick is already scheduled,
        which bounds the delay before the app actually starts closing.
        """
        self._interrupted = True

    def _on_close(self):
        """Stop the executor thread before tearing down the window.

        Reachable from the window's close button and from Ctrl-C, so it has to be idempotent.
        """
        if self._closing:
            return
        self._closing = True
        self._executor.shutdown(timeout_sec=0)
        self._spin_thread.join(timeout=1.0)
        self.root.destroy()

    def pid_publish(self, section):
        """Publish the PID values to the corresponding ROS2 topic based on the section."""
        if section == "throttle":
            self.throttle_pub.publish(self.pid_values(section))
        elif section == "pitch":
            self.pitch_pub.publish(self.pid_values(section))
        elif section == "roll":
            self.roll_pub.publish(self.pid_values(section))

    def pid_values(self, value):
        pid_msg = PIDTune()
        # Get values from the PIDTuningApp instance
        pid_msg.kp = float(int(self.entries[value + "_Kp"].get()) * self._get_multiplier(value, "Kp"))
        pid_msg.ki = float(int(self.entries[value + "_Ki"].get()) * self._get_multiplier(value, "Ki"))
        pid_msg.kd = float(int(self.entries[value + "_Kd"].get()) * self._get_multiplier(value, "Kd"))
        return pid_msg

    def _load_icons(self):
        """Load and set the icons."""
        package_dir = get_package_share_directory('pid_tune')
        self.left_icon_image = Image.open(os.path.join(package_dir, 'resources', 'e.png'))
        self.right_icon_image = Image.open(os.path.join(package_dir, 'resources', 'drone.webp'))
        self.success_image = Image.open(os.path.join(package_dir, 'resources', 'success.webp'))

        self.left_icon_image = self.left_icon_image.resize((30, 30), Image.LANCZOS)
        self.left_icon_photo = ImageTk.PhotoImage(self.left_icon_image)

        self.right_icon_image = self.right_icon_image.resize((44, 44), Image.LANCZOS)
        self.right_icon_photo = ImageTk.PhotoImage(self.right_icon_image)

        self.success_image = self.success_image.resize((18, 18), Image.LANCZOS)
        self.success_photo = ImageTk.PhotoImage(self.success_image)

    def _create_fonts(self):
        """Build the handful of shared fonts the whole UI draws with.

        Every widget reuses these objects instead of naming a family/size inline: Tk then
        allocates each font once. The families come from Tk's own defaults so they always
        exist -- asking for a family the system lacks (e.g. Arial on Ubuntu) sends every
        lookup through fontconfig fallback, which crashes Tk's font allocator on this build.
        """
        ui_family = tkfont.nametofont('TkDefaultFont').actual('family')
        mono_family = tkfont.nametofont('TkFixedFont').actual('family')
        self.fonts = {
            'title': tkfont.Font(family=ui_family, size=11, weight='bold'),
            'card': tkfont.Font(family=ui_family, size=10, weight='bold'),
            'label': tkfont.Font(family=ui_family, size=9, weight='bold'),
            'body': tkfont.Font(family=ui_family, size=9),
            'caption': tkfont.Font(family=ui_family, size=7),
            'value': tkfont.Font(family=mono_family, size=9, weight='bold'),
        }

    def _apply_responsive_geometry(self):
        """Derive the window's minimum size from what the widgets actually need.

        Tk clips content instead of scrolling it, so the resize floor is one card wide and tall
        enough for the stacked layout plus the plot: every gain stays readable at any size.

        The minimum must also stay below half a monitor, because the window manager refuses to
        edge-tile (drag-snap) a window whose minimum size hints don't fit in the tile area.
        """
        self.root.update_idletasks()
        card_w = max(card.winfo_reqwidth() for card in self._cards)
        card_h = max(card.winfo_reqheight() for card in self._cards)

        # Wide enough for one card and for the graph's control strip, which never wraps.
        min_w = max(card_w + 16, self.graph_controls.winfo_reqwidth() + 28, self.MIN_GRAPH_SIZE[0])
        # tuner_frame currently holds one row of cards; stacking them adds two more rows.
        min_h = (self.tuner_frame.winfo_reqheight() + 2 * (card_h + 6)
                 + self.graph_controls.winfo_reqheight() + self.MIN_GRAPH_SIZE[1] + 14)
        self.root.minsize(min_w, min_h)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(max(min_w, 1180), screen_w - 80)
        height = min(max(min_h, 720), screen_h - 80)
        self.root.geometry(f'{width}x{height}')

    def _create_widgets(self):
        """Create and place all widgets: the tuner on top (sections side by side), the error graph filling the bottom."""
        self.main_container = tk.Frame(self.root, bg=BG)
        self.main_container.pack(fill='both', expand=True)

        self.tuner_frame = tk.Frame(self.main_container, bg=BG)
        self.tuner_frame.pack(side='top', fill='x')

        self._create_header()
        self._create_sections()
        self._create_buttons()

        self.graph_frame = tk.Frame(self.main_container, bg=BG)
        self.graph_frame.pack(side='top', fill='both', expand=True)
        self._create_graph_panel()

    def _create_header(self):
        """Create the header row with icons, title and the Save control."""
        header = tk.Frame(self.tuner_frame, bg=BG)
        header.pack(fill='x', padx=10, pady=(5, 4))
        tk.Label(header, image=self.left_icon_photo, bg=BG).pack(side='left')
        tk.Label(header, text="Swift Pico PID Tuner", font=self.fonts['title'], bg=BG, fg=TEXT_LIGHT).pack(
            side='left', padx=(8, 0))
        tk.Label(header, image=self.right_icon_photo, bg=BG).pack(side='right')

    def _create_sections(self):
        """Create the Throttle, Pitch and Roll cards, laid out by _reflow_sections."""
        self.sections_row = tk.Frame(self.tuner_frame, bg=BG)
        self.sections_row.pack(fill='x', padx=8)

        self._cards = [self._create_section(self.sections_row, 'Throttle', 'throttle'),
                       self._create_section(self.sections_row, 'Pitch', 'pitch'),
                       self._create_section(self.sections_row, 'Roll', 'roll')]
        self._section_columns = 0
        # Lay them out three across first, so the startup geometry is measured against that.
        self.root.update_idletasks()
        self._reflow_sections(10 ** 6)
        self.sections_row.bind('<Configure>', self._on_sections_configure, True)

    def _on_sections_configure(self, event):
        """Re-flow the cards when the row's width changes."""
        self._reflow_sections(event.width)

    def _reflow_sections(self, width):
        """Fit as many cards per row as the width allows: 3 across when wide, stacking as it narrows.

        This is what lets the window shrink to a single card's width, which in turn keeps the
        window's minimum size small enough for the window manager to drag-snap it to a screen half.
        """
        card_w = max(card.winfo_reqwidth() for card in self._cards) + 8
        columns = max(1, min(len(self._cards), width // card_w))
        if columns == self._section_columns:
            return
        self._section_columns = columns

        for i in range(len(self._cards)):
            self.sections_row.grid_columnconfigure(
                i, weight=1 if i < columns else 0, uniform='section' if i < columns else '')
        for index, card in enumerate(self._cards):
            card.grid(row=index // columns, column=index % columns, padx=4, pady=(0, 6), sticky='nsew')

    def _create_section(self, parent, title, prefix):
        """Create one axis card: a caption row plus a self-contained Kp/Ki/Kd row each."""
        card = tk.Frame(parent, bg=CARD_BG, bd=0)

        tk.Label(card, text=f'● {title.upper()}', bg=CARD_BG, fg=ACCENT, font=self.fonts['card']).grid(
            row=0, column=0, columnspan=8, padx=8, pady=(6, 1), sticky='w')

        # Tiny column captions, so each row reads as: gain  −[value]+  ×scale  = result  step
        for text, col, span in (('value', 1, 3), ('scale', 5, 1), ('result', 6, 1), ('step', 7, 1)):
            tk.Label(card, text=text, bg=CARD_BG, fg=MUTED, font=self.fonts['caption']).grid(
                row=1, column=col, columnspan=span, pady=(0, 1))

        for row, key in enumerate(('Kp', 'Ki', 'Kd'), start=2):
            self._create_pid_row(card, key, row, prefix)
        tk.Frame(card, bg=CARD_BG, height=3).grid(row=5, column=0, columnspan=8)

        # Collect spare width in a trailing spacer column so the controls stay grouped
        # together instead of being spread across the card as the window widens.
        card.grid_columnconfigure(8, weight=1)

        for key in ('Kp', 'Ki', 'Kd'):
            self._refresh_computed(prefix, key)
        return card

    def _create_pid_row(self, parent, label_text, row, prefix):
        """Create one gain row: label, -/+ stepper, its own scale multiplier, the scaled result and step size."""
        name = f'{prefix}_{label_text}'

        tk.Label(parent, text=label_text, bg=CARD_BG, fg=TEXT_LIGHT, font=self.fonts['label']).grid(
            row=row, column=0, padx=(8, 3), pady=2, sticky='w')

        tk.Button(parent, text='−', width=2, font=self.fonts['label'],
                  command=lambda: self._decrease_value(self.entries[name], self.step_entries[f'{name}_step'], prefix),
                  bg=ACCENT, fg=TEXT_LIGHT, relief='flat', bd=0,
                  activebackground=_darken(ACCENT), activeforeground=TEXT_LIGHT).grid(row=row, column=1, padx=1)

        # The multiplier must exist before _initial_value(), which converts the saved
        # (already scaled) gain back into the raw integer shown in the entry.
        mult_var = tk.StringVar(value=str(self.MULT_DEFAULTS[label_text]))
        mult_entry = tk.Entry(parent, textvariable=mult_var, width=5, bg=BG, fg=TEXT_LIGHT, justify='center',
                              relief='flat', bd=1, highlightthickness=1, highlightbackground=MUTED,
                              insertbackground=TEXT_LIGHT, font=self.fonts['body'])
        self.multiplier_entries[name] = mult_entry
        self._mult_vars[name] = mult_var

        value_var = tk.StringVar(value=self._initial_value(prefix, label_text))
        entry = tk.Entry(parent, textvariable=value_var, width=4, bg=BG, fg=TEXT_LIGHT, justify='center',
                         relief='flat', bd=1, highlightthickness=1, highlightbackground=MUTED,
                         insertbackground=TEXT_LIGHT, font=self.fonts['body'])
        entry.grid(row=row, column=2, padx=1)
        entry.bind('<Return>', lambda event, p=prefix: self.pid_publish(p))
        self.entries[name] = entry
        self._value_vars[name] = value_var
        value_var.trace_add('write', lambda *_, p=prefix, k=label_text: self._refresh_computed(p, k))

        tk.Button(parent, text='+', width=2, font=self.fonts['label'],
                  command=lambda: self._increase_value(entry, self.step_entries[f'{name}_step'], prefix),
                  bg=ACCENT, fg=TEXT_LIGHT, relief='flat', bd=0,
                  activebackground=_darken(ACCENT), activeforeground=TEXT_LIGHT).grid(row=row, column=3, padx=1)

        tk.Label(parent, text='×', bg=CARD_BG, fg=MUTED, font=self.fonts['body']).grid(row=row, column=4, padx=(4, 0))
        mult_entry.grid(row=row, column=5, padx=(0, 4))
        mult_var.trace_add('write', lambda *_, p=prefix, k=label_text: self._refresh_computed(p, k))

        computed_label = tk.Label(parent, text='= 0.000', bg=CARD_BG, fg=ACCENT, font=self.fonts['value'],
                                  width=9, anchor='w')
        computed_label.grid(row=row, column=6, padx=(2, 4))
        self.computed_labels[name] = computed_label

        step_entry = tk.Entry(parent, width=3, bg=BG, fg=TEXT_LIGHT, justify='center', relief='flat', bd=1,
                              highlightthickness=1, highlightbackground=MUTED, insertbackground=TEXT_LIGHT,
                              font=self.fonts['body'])
        step_entry.grid(row=row, column=7, padx=(0, 8))
        step_entry.insert(0, "1")
        self.step_entries[f'{name}_step'] = step_entry

    def _refresh_computed(self, prefix, key):
        """Update the live '= scaled value' label for one Kp/Ki/Kd row."""
        label = self.computed_labels.get(f'{prefix}_{key}')
        if label is None:
            return
        try:
            raw = int(self.entries[f'{prefix}_{key}'].get())
            mult = float(self.multiplier_entries[f'{prefix}_{key}'].get())
        except ValueError:
            label.config(text='= —')
            return
        label.config(text=f'= {raw * mult:.3f}')

    def _create_buttons(self):
        """Create the Save button and success icon, centered below the three tuning sections."""
        button_frame = tk.Frame(self.tuner_frame, bg=BG)
        button_frame.pack(pady=(0, 6))

        tk.Button(button_frame, text='Save Values', font=self.fonts['label'], command=self._save_values,
                  bg=ACCENT, fg=TEXT_LIGHT, relief='flat', bd=0, padx=14, pady=4,
                  activebackground=_darken(ACCENT), activeforeground=TEXT_LIGHT).grid(row=0, column=0, padx=(0, 8))

        self.success_label = tk.Label(button_frame, image=self.success_photo, bg=BG)
        self.success_label.grid(row=0, column=1, padx=(8, 0))
        self.success_label.grid_remove()

    def _create_graph_panel(self):
        """Create the /pos_error live-graph panel: one shared plot, a series picker, play/pause and window controls."""
        controls = self.graph_controls = tk.Frame(self.graph_frame, bg=BG)
        controls.pack(fill='x', padx=14, pady=(0, 4))

        tk.Label(controls, text='● /pos_error', bg=BG, fg=ACCENT, font=self.fonts['card']).pack(
            side='left', padx=(0, 14))

        self.play_pause_btn = tk.Button(controls, text='⏸ Pause', width=8, font=self.fonts['label'],
                                         command=self._toggle_pause, bg=ACCENT, fg=TEXT_LIGHT, relief='flat', bd=0,
                                         activebackground=_darken(ACCENT), activeforeground=TEXT_LIGHT)
        self.play_pause_btn.pack(side='left')

        tk.Label(controls, text='Window', bg=BG, fg=MUTED, font=self.fonts['body']).pack(side='left', padx=(16, 4))
        tk.Button(controls, text='▼', width=2, font=self.fonts['label'], command=lambda: self._adjust_window(-1),
                  bg=ACCENT, fg=TEXT_LIGHT, relief='flat', bd=0,
                  activebackground=_darken(ACCENT), activeforeground=TEXT_LIGHT).pack(side='left')
        self.window_seconds = tk.IntVar(value=10)
        tk.Label(controls, textvariable=self.window_seconds, width=3, bg=CARD_BG, fg=TEXT_LIGHT,
                 font=self.fonts['value']).pack(side='left', padx=3)
        tk.Button(controls, text='▲', width=2, font=self.fonts['label'], command=lambda: self._adjust_window(1),
                  bg=ACCENT, fg=TEXT_LIGHT, relief='flat', bd=0,
                  activebackground=_darken(ACCENT), activeforeground=TEXT_LIGHT).pack(side='left')
        tk.Label(controls, text='s', bg=BG, fg=MUTED, font=self.fonts['body']).pack(side='left', padx=(3, 0))

        self._series_vars = {}
        for key, (_, color, label) in self.SERIES.items():
            var = tk.BooleanVar(value=True)
            self._series_vars[key] = var
            tk.Checkbutton(controls, text=label, variable=var, command=self._on_series_toggle,
                           bg=BG, fg=color, selectcolor=CARD_BG, activebackground=BG, activeforeground=color,
                           font=self.fonts['label'], relief='flat', highlightthickness=0).pack(
                side='left', padx=(14, 0))

        # Blit state: the static background (grid/ticks/labels) is cached and only re-rendered
        # when the axes limits, the canvas size or the time window actually change.
        self._bg = None
        self._cached_ylim = None
        self._shrink_since = None
        self._drawn_window = None
        self._last_sample_t = None
        self._resize_job = None

        self.fig = Figure(figsize=(9, 4.5), dpi=100, facecolor=CARD_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(CARD_BG)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        self.ax.grid(True, color='#2a2d33', linewidth=0.5)
        for spine in self.ax.spines.values():
            spine.set_color(MUTED)
        self.ax.set_xlabel('seconds ago', color=MUTED, fontsize=8)
        self.ax.set_xlim(-self.window_seconds.get(), 0)
        self.ax.set_ylim(-self.Y_MIN_SPAN, self.Y_MIN_SPAN)
        self.ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.6)

        # No legend: the series checkboxes above are colored to match their traces.
        self._graph_lines = {}
        for key, (_, color, _label) in self.SERIES.items():
            line, = self.ax.plot([], [], color=color, linewidth=1.4, animated=True)
            self._graph_lines[key] = line

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True, padx=14, pady=(0, 10))
        # add=True: matplotlib binds <Configure> to its own resize handler, which must keep running.
        canvas_widget.bind('<Configure>', self._on_canvas_resize, True)

    def _adjust_window(self, delta):
        """Step the plotted time window by delta seconds, clamped to 1..MAX_WINDOW_SECONDS."""
        current = self.window_seconds.get()
        new_value = max(1, min(self.MAX_WINDOW_SECONDS, current + delta))
        if new_value != current:
            self.window_seconds.set(new_value)

    def _on_canvas_resize(self, _event):
        """Invalidate the cached background on resize, debounced so dragging doesn't trigger a redraw storm."""
        self._bg = None
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self._refresh_after_resize)

    def _refresh_after_resize(self):
        """Rebuild the background and repaint the traces once resizing has settled."""
        self._resize_job = None
        self._redraw_background()
        self._blit_lines()

    def _apply_figure_margins(self):
        """Reserve a constant number of pixels for the axis labels.

        Fractional margins would shrink with the figure and squeeze the tick labels out at
        small window sizes; in pixels the plot stays readable however small the window gets.
        """
        width, height = self.canvas.get_width_height()
        if width <= 0 or height <= 0:
            return
        self.fig.subplots_adjust(left=min(52 / width, 0.30), right=1 - min(12 / width, 0.10),
                                 bottom=min(34 / height, 0.30), top=1 - min(10 / height, 0.12))

    def _redraw_background(self):
        """Full (expensive) figure render, then cache the static background for later blits."""
        self._apply_figure_margins()
        self.canvas.draw()
        self._bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def _blit_lines(self):
        """Cheap redraw: restore the cached background and re-blit only the visible traces."""
        if self._bg is None:
            self._redraw_background()
        self.canvas.restore_region(self._bg)
        for line in self._graph_lines.values():
            if line.get_visible():
                self.ax.draw_artist(line)
        self.canvas.blit(self.ax.bbox)

    def _on_series_toggle(self):
        """Show/hide traces immediately when a checkbox changes, without waiting for the next sample."""
        for key, var in self._series_vars.items():
            self._graph_lines[key].set_visible(var.get())
        self._cached_ylim = None  # rescale to whatever is still on screen
        self._shrink_since = None
        self._last_sample_t = None  # let the next tick repaint
        if self.paused:
            self._blit_lines()

    def _toggle_pause(self):
        """Pause/resume the graph display without dropping incoming /pos_error samples."""
        self.paused = not self.paused
        self.play_pause_btn.config(text='▶ Play' if self.paused else '⏸ Pause')

    def _pos_error_callback(self, msg):
        """Buffer an incoming /pos_error sample, trimming anything older than MAX_WINDOW_SECONDS."""
        t = time.time() - self.start_time
        cutoff = t - self.MAX_WINDOW_SECONDS
        with self._data_lock:
            self.time_buf.append(t)
            self.throttle_buf.append(msg.throttle_error)
            self.pitch_buf.append(msg.pitch_error)
            self.roll_buf.append(msg.roll_error)

            while self.time_buf and self.time_buf[0] < cutoff:
                self.time_buf.popleft()
                self.throttle_buf.popleft()
                self.pitch_buf.popleft()
                self.roll_buf.popleft()

    @staticmethod
    def _nice_step(raw):
        """Round raw up to the next 1/2/5 x 10^n, so the axis lands on readable values."""
        exponent = math.floor(math.log10(raw))
        fraction = raw / 10 ** exponent
        nice = 1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0
        return nice * 10 ** exponent

    def _autoscale_y(self, visible_ys):
        """Fit the y axis to the amplitude on screen: zoom in as the error settles, out as it grows.

        The limits are snapped outward to a 1/2/5 grid step, so they only move when the trace
        actually crosses a step rather than drifting with every sample, and zooming in waits
        Y_SHRINK_HOLD_S so one calm patch doesn't make the axis jump. Returns True when the
        limits changed, meaning the cached background has to be rebuilt.
        """
        if visible_ys:
            low, high = min(visible_ys), max(visible_ys)
        else:
            low = high = 0.0
        # Keep zero on screen: on an error trace the distance to zero is what is being read.
        low, high = min(low, 0.0), max(high, 0.0)
        pad = (high - low) * self.Y_PAD
        step = self._nice_step(max(high - low + 2 * pad, self.Y_MIN_SPAN) / self.Y_TICK_TARGET)
        lo = math.floor((low - pad) / step) * step
        hi = math.ceil((high + pad) / step) * step
        while hi - lo < self.Y_MIN_SPAN:  # a perfectly flat trace would otherwise collapse the axis
            lo, hi = lo - step, hi + step
        target = (lo, hi)

        current = self._cached_ylim
        if current is not None and current[0] <= target[0] and target[1] <= current[1]:
            if (current[1] - current[0]) <= (target[1] - target[0]) * self.Y_SHRINK_RATIO:
                self._shrink_since = None  # the axis already fits the trace closely enough
                return False
            now = time.monotonic()
            if self._shrink_since is None:
                self._shrink_since = now
            if now - self._shrink_since < self.Y_SHRINK_HOLD_S:
                return False  # wait out the hold before zooming in

        self._shrink_since = None
        self._cached_ylim = target
        self.ax.set_ylim(*target)
        return True

    def _update_graph(self):
        """Render tick: refresh the plot only when a new sample (or a new window size) means it changed."""
        self.root.after(self.GRAPH_UPDATE_MS, self._update_graph)

        if self._interrupted:  # Ctrl-C arrived: close from the Tk thread, not the signal handler
            self._on_close()
            return

        if self.paused:
            return

        window = self.window_seconds.get()
        with self._data_lock:
            if not self.time_buf:
                return
            t_now = self.time_buf[-1]
            if t_now == self._last_sample_t and window == self._drawn_window:
                return  # nothing new arrived since the last frame
            times = list(self.time_buf)
            samples = {key: list(getattr(self, buf_name)) for key, (buf_name, _, _) in self.SERIES.items()}
        self._last_sample_t = t_now

        cutoff = t_now - window
        start_idx = len(times) - 1
        for i in range(len(times) - 1, -1, -1):
            if times[i] < cutoff:
                break
            start_idx = i

        # Plot against time-relative-to-now so the x axis (and therefore the cached
        # background) stays fixed at [-window, 0] instead of moving every frame.
        xs = [t - t_now for t in times[start_idx:]]
        visible_ys = []
        for key in self.SERIES:
            ys = samples[key][start_idx:]
            self._graph_lines[key].set_data(xs, ys)
            if self._series_vars[key].get():
                visible_ys.extend(ys)

        needs_background = self._bg is None

        if window != self._drawn_window:
            self.ax.set_xlim(-window, 0)
            self._drawn_window = window
            needs_background = True

        if self._autoscale_y(visible_ys):
            needs_background = True

        if needs_background:
            self._redraw_background()
        self._blit_lines()

    def _validate_integer(self, value):
        """Check if the value is a positive integer."""
        try:
            int_value = int(value)
            if int_value < 0:
                return False
            return True
        except ValueError:
            return False

    def _increase_value(self, entry, step_entry, section):
        """Increase the value in the entry widget and publish the PID values."""
        entry_value = entry.get()
        step_value = step_entry.get()

        if not self._validate_integer(entry_value):
            messagebox.showerror("Invalid Input", "Please enter a positive integer for the value.")
            return

        if not self._validate_integer(step_value):
            messagebox.showerror("Invalid Input", "Please enter a positive integer for the step size.")
            return

        current_value = int(entry_value)
        step_size = int(step_value)
        entry.delete(0, tk.END)
        entry.insert(0, str(current_value + step_size))

        # Publish the updated values
        self.pid_publish(section)

    def _decrease_value(self, entry, step_entry, section):
        """Decrease the value in the entry widget and publish the PID values."""
        entry_value = entry.get()
        step_value = step_entry.get()

        if not self._validate_integer(entry_value):
            messagebox.showerror("Invalid Input", "Please enter a positive integer for the value.")
            return

        if not self._validate_integer(step_value):
            messagebox.showerror("Invalid Input", "Please enter a positive integer for the step size.")
            return

        current_value = int(entry_value)
        step_size = int(step_value)
        new_value = current_value - step_size
        if new_value < 0:
            new_value = 0
        entry.delete(0, tk.END)
        entry.insert(0, str(new_value))

        # Publish the updated values
        self.pid_publish(section)

    def _find_scripts_dir(self):
        """Locate src/controller_tuner/pid_tune/scripts by walking up from this file.

        This node runs from the built install/ copy, so __file__'s own directory
        isn't the source tree. Fall back to it if the workspace src/ can't be found
        (e.g. package installed without a sibling src/ checkout).
        """
        current = os.path.dirname(os.path.abspath(__file__))
        while True:
            # candidate = os.path.join(current, 'src', 'controller_tuner', 'pid_tune', 'scripts')
            candidate = os.path.join(current, 'src', 'swift_pico', 'src') # for Swift Pico package and pico_ws workspace
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                return os.path.dirname(os.path.abspath(__file__))
            current = parent

    def _load_saved_values(self):
        """Load previously saved PID values from pid_values.yaml, if it exists."""
        yaml_path = os.path.join(self._find_scripts_dir(), 'pid_values.yaml')
        if not os.path.isfile(yaml_path):
            return {}
        try:
            with open(yaml_path, 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)
        except (yaml.YAMLError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _initial_value(self, prefix, key):
        """Resume from the last saved (scaled) value for prefix/key, or '0' if none was saved."""
        section = self._saved_values.get(f'{prefix}_pid', {})
        saved_scaled = section.get('ros__parameters', {}).get(key) if isinstance(section, dict) else None
        if saved_scaled is None:
            return "0"
        mult = float(self._mult_vars[f'{prefix}_{key}'].get())
        if mult <= 0:
            return "0"
        return str(round(saved_scaled / mult))

    def _get_multiplier(self, prefix, key):
        """Read and validate a section's save-multiplier entry for Kp/Ki/Kd."""
        raw = self.multiplier_entries[f'{prefix}_{key}'].get()
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{prefix.title()} {key} multiplier must be a number.")
        if value <= 0:
            raise ValueError(f"{prefix.title()} {key} multiplier must be positive.")
        return value

    def _scaled_section_values(self, prefix):
        """Read a section's raw Kp/Ki/Kd entries and scale them by that section's multipliers."""
        values = {}
        for key in ('Kp', 'Ki', 'Kd'):
            raw = int(self.entries[f"{prefix}_{key}"].get())
            if raw < 0:
                raise ValueError("Values must be positive integers.")
            values[key] = raw * self._get_multiplier(prefix, key)
        return values

    def _save_values(self):
        """Save PID values (scaled by each section's multipliers) to a file."""
        try:
            throttle_values = self._scaled_section_values('throttle')
            pitch_values = self._scaled_section_values('pitch')
            roll_values = self._scaled_section_values('roll')

            # Structure the data to match a typical ROS 2 parameter YAML file format
            yaml_data = {
                'throttle_pid': {
                    'ros__parameters': throttle_values
                },
                'pitch_pid': {
                    'ros__parameters': pitch_values
                },
                'roll_pid': {
                    'ros__parameters': roll_values
                }
            }

            # Write the YAML data to the source scripts dir (not the install copy this runs from),
            # so it survives rebuilds and is easy to find in the workspace.
            yaml_path = os.path.join(self._find_scripts_dir(), 'pid_values.yaml')
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)

            self.get_logger().info(f"Saved PID values to {yaml_path}")
            self.success_label.grid()  # Show the success icon
            self.root.after(2000, self._hide_success_icon)  # Hide the success icon after 2 seconds

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def _hide_success_icon(self):
        """Hide the success icon."""
        self.success_label.grid_remove()


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = PIDTuningApp(root)
    try:
        root.mainloop()
    finally:
        app.destroy_node()
        if rclpy.ok():  # an external shutdown may already have closed the context
            rclpy.shutdown()


if __name__ == '__main__':
    main()
