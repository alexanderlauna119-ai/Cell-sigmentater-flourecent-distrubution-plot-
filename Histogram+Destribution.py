
# ===================== IMPORTS =====================

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.colorchooser import askcolor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.widgets import Button


# ===================== APP =====================

class OverlayHistogramApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Fluorescence Population Analysis")

        self.excel = None
        self.sheet_vars = {}
        self.sheet_colors = {}

        self.column_var = tk.StringVar()
        self.control_var = tk.StringVar()
        self.gate_mode_var = tk.StringVar(value="Low + High")

        self.mfi_df = None
        self.pop_df = None

        # -------- GUI --------

        tk.Button(root, text="Choose Excel file",
                  command=self.load_file).pack(pady=6)

        tk.Label(root, text="Select sheets").pack()
        self.sheet_frame = tk.Frame(root)
        self.sheet_frame.pack()

        # ---- Order control (Figure 2) ----
        tk.Label(root, text="Treatment order (Figure 2)").pack(pady=4)

        order_frame = tk.Frame(root)
        order_frame.pack()

        self.order_listbox = tk.Listbox(order_frame, width=30, height=6)
        self.order_listbox.pack(side="left")

        btn_frame = tk.Frame(order_frame)
        btn_frame.pack(side="left", padx=5)

        tk.Button(btn_frame, text="▲ Up",
                  command=self.move_up).pack(fill="x")
        tk.Button(btn_frame, text="▼ Down",
                  command=self.move_down).pack(fill="x")

        tk.Label(root, text="Negative control").pack()
        self.control_menu = tk.OptionMenu(root, self.control_var, "")
        self.control_menu.pack()

        tk.Label(root, text="Fluorescence column").pack()
        self.column_menu = tk.OptionMenu(root, self.column_var, "")
        self.column_menu.pack()

        tk.Label(root, text="Gates to display (Figure 1)").pack()
        tk.OptionMenu(
            root,
            self.gate_mode_var,
            "None",
            "Low only",
            "High only",
            "Low + High"
        ).pack(pady=4)

        tk.Button(root, text="Plot + Analyze",
                  command=self.plot).pack(pady=10)

    # ------------------------------------------------

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not path:
            return

        self.excel = pd.ExcelFile(path)

        for w in self.sheet_frame.winfo_children():
            w.destroy()

        self.sheet_vars.clear()
        self.sheet_colors.clear()
        self.order_listbox.delete(0, "end")

        for sheet in self.excel.sheet_names:
            row = tk.Frame(self.sheet_frame)
            row.pack(anchor="w")

            var = tk.BooleanVar()
            tk.Checkbutton(row, text=sheet,
                           variable=var).pack(side="left")

            tk.Button(row, text="Color",
                      command=lambda s=sheet: self.pick_color(s)
                      ).pack(side="left")

            self.sheet_vars[sheet] = var
            self.sheet_colors[sheet] = None

            self.order_listbox.insert("end", sheet)

        # Control selector
        self.control_var.set(self.excel.sheet_names[0])
        self.control_menu["menu"].delete(0, "end")
        for s in self.excel.sheet_names:
            self.control_menu["menu"].add_command(
                label=s,
                command=lambda x=s: self.control_var.set(x)
            )

        # Column selector
        df = pd.read_excel(self.excel,
                           sheet_name=self.excel.sheet_names[0])
        cols = list(df.columns)

        self.column_var.set(cols[0])
        self.column_menu["menu"].delete(0, "end")
        for c in cols:
            self.column_menu["menu"].add_command(
                label=c,
                command=lambda x=c: self.column_var.set(x)
            )

    # ------------------------------------------------

    def move_up(self):
        sel = self.order_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        text = self.order_listbox.get(i)
        self.order_listbox.delete(i)
        self.order_listbox.insert(i - 1, text)
        self.order_listbox.select_set(i - 1)

    def move_down(self):
        sel = self.order_listbox.curselection()
        if not sel or sel[0] == self.order_listbox.size() - 1:
            return
        i = sel[0]
        text = self.order_listbox.get(i)
        self.order_listbox.delete(i)
        self.order_listbox.insert(i + 1, text)
        self.order_listbox.select_set(i + 1)

    # ------------------------------------------------

    def pick_color(self, sheet):
        color = askcolor(title=f"{sheet} color")
        if color and color[1]:
            self.sheet_colors[sheet] = color[1]

    # ------------------------------------------------

    def plot(self):

        selected = [s for s, v in self.sheet_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Error",
                                   "Select at least one sheet")
            return

        control = self.control_var.get()
        if control not in selected:
            messagebox.showwarning(
                "Error",
                "Negative control must be selected"
            )
            return

        # ---- Apply GUI order ----
        gui_order = list(self.order_listbox.get(0, "end"))
        plot_order = [s for s in gui_order if s in selected]

        # ---------- Load data ----------
        data = {}
        for sheet in plot_order:
            df = pd.read_excel(self.excel, sheet_name=sheet)
            vals = pd.to_numeric(
                df[self.column_var.get()],
                errors="coerce"
            ).dropna()
            vals = vals[vals > 0]
            if len(vals):
                data[sheet] = vals

        # ---------- Gating ----------
        log_ctrl = np.log10(data[control])
        mu = log_ctrl.mean()
        sd = log_ctrl.std()

        thr_low = 10 ** (mu + 2 * sd)
        thr_high = 10 ** (mu + 3.75 * sd)

        # ================= FIGURE 1 =================
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        mfi_rows = []
        pop_rows = []

        for cond, vals in data.items():
            bins = np.logspace(
                np.log10(vals.min()),
                np.log10(vals.max()),
                120
            )

            counts, edges = np.histogram(vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2

            ax1.plot(
                centers,
                counts,
                lw=2,
                color=self.sheet_colors.get(cond)
            )

            mfi_rows.append({
                "Condition": cond,
                "N": len(vals),
                "MFI": vals.mean()
            })

            pop_rows.append({
                "Condition": cond,
                "Low": np.sum(vals < thr_low),
                "Medium": np.sum(
                    (vals >= thr_low) & (vals < thr_high)
                ),
                "High": np.sum(vals >= thr_high)
            })

        self.mfi_df = pd.DataFrame(mfi_rows).set_index("Condition")
        self.pop_df = pd.DataFrame(pop_rows).set_index("Condition")

        ax1.set_xscale("log")
        ax1.set_xlabel("Fluorescence intensity")
        ax1.set_ylabel("Cell count")
        ax1.set_title("Fluorescence distributions")
        ax1.grid(alpha=0.2)

        if self.gate_mode_var.get() in ("Low only", "Low + High"):
            ax1.axvline(thr_low, color="black", ls="--")
        if self.gate_mode_var.get() in ("High only", "Low + High"):
            ax1.axvline(thr_high, color="black", ls=":")

        # ---- Custom legend with N and MFI (RESULTS) ----
        handles = [
            Line2D(
                [0], [0],
                lw=3,
                color=self.sheet_colors.get(cond),
                label=(
                    f"{cond} | "
                    f"N={int(self.mfi_df.loc[cond,'N'])} | "
                    f"MFI={self.mfi_df.loc[cond,'MFI']:,.1f}"
                )
            )
            for cond in self.mfi_df.index
        ]

        ax1.legend(handles=handles, loc="upper left")

        fig1.tight_layout()
        plt.show(block=False)

        # ================= FIGURE 2 =================
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        df_bar = self.pop_df.loc[plot_order]
        df_pct = df_bar.div(df_bar.sum(axis=1), axis=0) * 100

        x = np.arange(len(df_pct))
        w = 0.25

        ax2.bar(x - w, df_pct["Low"], w, color="gray", label="Low")
        ax2.bar(x, df_pct["Medium"], w, color="orange", label="Medium")
        ax2.bar(x + w, df_pct["High"], w, color="red", label="High")

        ax2.set_xticks(x)
        ax2.set_xticklabels(plot_order,
                             rotation=45, ha="right")

        ax2.set_ylabel("Percentage of cells (%)")
        ax2.set_ylim(0, 100)
        ax2.set_title("Cell population distribution")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend()

        fig2.tight_layout()
        plt.show(block=False)


# ===================== MAIN =====================

if __name__ == "__main__":
    root = tk.Tk()
    app = OverlayHistogramApp(root)
    root.mainloop()