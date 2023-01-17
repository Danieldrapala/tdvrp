from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import csv
import os

from simulated_annealing import simulated_annealing, Point


class TreeviewEdit(ttk.Treeview):

    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.bind("<Double-1>", self.on_double_click)

    def on_double_click(self, event):
        # Identify the region that was double-clicked
        region_clicked = self.identify_region(event.x, event.y)

        # We're only intrested in tree and cells
        if region_clicked not in ("tree", "cell"):
            return

        # Which item was double-clicked?
        column = self.identify_column(event.x)
        column_index = int(column[1:]) - 1

        selected_iid = self.focus()
        selected_values = self.item(selected_iid)

        selected_text = selected_values.get("values")[column_index]

        column_box = self.bbox(selected_iid, column)

        entry_edit = ttk.Entry(root)

        # Record the column index and item iid
        entry_edit.editing_column_index = column_index
        entry_edit.editing_item_iid = selected_iid

        entry_edit.insert(0, selected_text)
        entry_edit.select_range(0, tk.END)

        entry_edit.focus()

        entry_edit.bind("<FocusOut>", self.on_focus_out)
        entry_edit.bind("<Return>", self.on_enter_pressed)

        entry_edit.place(x=column_box[0] + 70,
                         y=column_box[1],
                         w=column_box[2],
                         h=column_box[3])

    def on_focus_out(self, event):
        event.widget.destroy()

    def on_enter_pressed(self, event):
        new_text = event.widget.get()

        selected_iid = event.widget.editing_item_iid

        column_index = event.widget.editing_column_index

        current_values = self.item(selected_iid).get("values")
        current_values[column_index] = new_text
        self.item(selected_iid, values=current_values)

        event.widget.destroy()


if __name__ == "__main__":

    mydata1 = []
    mydata2 = []


    def update_trv1(rows):
        global mydata1
        mydata1 = rows
        trv1.delete(*trv1.get_children())
        for i in rows:
            trv1.insert('', 'end', values=i)


    def update_trv2(rows):
        global mydata2
        mydata2 = rows
        trv2.delete(*trv2.get_children())
        for i in rows:
            trv2.insert('', 'end', values=i)


    def importcsv1():
        mydata1.clear()
        fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open CSV",
                                         filetypes=(("CSV File", "*.csv"), ("All FIles", "*.*")))
        with open(fln) as myfile:
            csvread = csv.reader(myfile, delimiter=",")
            for i in csvread:
                mydata1.append(i)
        update_trv1(mydata1)


    def importcsv2():
        mydata2.clear()
        fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open CSV",
                                         filetypes=(("CSV File", "*.csv"), ("All FIles", "*.*")))
        with open(fln) as myfile:
            csvread = csv.reader(myfile, delimiter=",")
            for i in csvread:
                mydata2.append(i)
        update_trv2(mydata2)


    def export1():
        if len(mydata1) < 1:
            messagebox.showerror("No Data", "No data available to export")
            return False

        fln = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save SCV",
                                           filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")))
        with open(fln, 'w', newline='') as myfile:
            exp_wrtier = csv.writer(myfile, delimiter=',')
            for i in mydata1:
                exp_wrtier.writerow(i)
        messagebox.showinfo("Data Exported",
                            "Your data has been exported to " + os.path.basename(fln) + " successfully.")


    def export2():
        if len(mydata2) < 1:
            messagebox.showerror("No Data", "No data available to export")
            return False

        fln = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save SCV",
                                           filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")))
        with open(fln, 'w', newline='') as myfile:
            exp_wrtier = csv.writer(myfile, delimiter=',')
            for i in mydata2:
                exp_wrtier.writerow(i)
        messagebox.showinfo("Data Exported",
                            "Your data has been exported to " + os.path.basename(fln) + " successfully.")


    root = Tk()

    wrapper1 = LabelFrame(root, text="Initial Data")
    wrapper2 = LabelFrame(root, text="Client Data")
    wrapper3 = LabelFrame(root, text="Time Windows")

    wrapper1.grid(row=0, column=0, padx="5")
    wrapper2.grid(row=0, column=1, padx="5", rowspan="100")
    wrapper3.grid(row=0, column=2, padx="5", rowspan="100")

    # INITIAL DATA
    # M=1 -stala
    Label(wrapper1, text="P parameter").grid(row=0, column=0, padx="5", pady="5")
    Label(wrapper1, text="MPG").grid(row=1, column=0, padx="5", pady="5")
    Label(wrapper1, text="Capacity").grid(row=2, column=0, padx="5", pady="5")
    Label(wrapper1, text="Pause").grid(row=3, column=0, padx="5", pady="5")
    Label(wrapper1, text="First departure time").grid(row=4, column=0, padx="5",
                                                      pady="5")

    a1 = Entry(wrapper1)
    a1.grid(row=0, column=1, padx="5", pady="5")
    b1 = Entry(wrapper1)
    b1.grid(row=1, column=1, padx="5", pady="5")
    c1 = Entry(wrapper1)
    c1.grid(row=2, column=1, padx="5", pady="5")
    d1 = Entry(wrapper1)
    d1.grid(row=3, column=1, padx="5", pady="5")
    e1 = Entry(wrapper1)
    e1.grid(row=4, column=1, padx="5", pady="5")

    # CUSTOMERS DATA LOOKUP
    trv1 = TreeviewEdit(wrapper2, columns=(1, 2, 3, 4, 5), show="headings", height="30")
    trv1.grid(row=0, column=0, columnspan=25, padx="5", pady="5")

    trv1.heading(1, text="Customer ID")
    trv1.heading(2, text="Name")
    trv1.heading(3, text="X")
    trv1.heading(4, text="Y")
    trv1.heading(5, text="Demand")
    trv1.column("# 1", anchor=CENTER, stretch=NO, width=100)
    trv1.column("# 2", anchor=CENTER, stretch=NO, width=100)
    trv1.column("# 3", anchor=CENTER, stretch=NO, width=100)
    trv1.column("# 4", anchor=CENTER, stretch=NO, width=100)
    trv1.column("# 5", anchor=CENTER, stretch=NO, width=100)

    impbtn1 = Button(wrapper2, text="Import CSV", command=importcsv1)
    impbtn1.grid(row=5, column=0, sticky="w", padx="5", pady="5")

    expbtn1 = Button(wrapper2, text="Export CSV", command=export1)
    expbtn1.grid(row=5, column=1, sticky="w", padx="5", pady="5")

    # TIME WINDOWS DATA LOOKUP
    trv2 = TreeviewEdit(wrapper3, columns=(1, 2, 3, 4), show="headings", height="30")
    trv2.grid(row=0, column=0, columnspan=25, padx="5", pady="5")

    trv2.heading(1, text="ID")
    trv2.heading(2, text="Starting Hour")
    trv2.heading(3, text="Ending Hour")
    trv2.heading(4, text="Velocity")
    trv2.column("# 1", anchor=CENTER, stretch=NO, width=100)
    trv2.column("# 2", anchor=CENTER, stretch=NO, width=100)
    trv2.column("# 3", anchor=CENTER, stretch=NO, width=100)
    trv2.column("# 4", anchor=CENTER, stretch=NO, width=100)

    impbtn2 = Button(wrapper3, text="Import CSV", command=importcsv2)
    impbtn2.grid(row=5, column=0, sticky="w", padx="5", pady="5")

    expbtn2 = Button(wrapper3, text="Export CSV", command=export2)
    expbtn2.grid(row=5, column=1, sticky="w", padx="5", pady="5")

    # OTHER BUTTONS
    extbtn = Button(master=root, text="Exit", width="10", command=lambda: exit())
    extbtn.grid(row=4, column=0, padx="5", pady="5", sticky="w")


    def makeRetailStoriesList():
        array = []
        for i in mydata1:
            array.append(Point((float(i[2]),float(i[3])),0,float(i[4])))
        return array


    def takeVelocityAndIntervalArray():
        V = []
        W = []
        for i in mydata2:
            V.append(float(i[3]))
            W.append(float(i[2]))
        return V,W

    def calc():
        print(mydata1)
        print(mydata2)
        print(a1.get())
        print(b1.get())
        print(c1.get())
        print(d1.get())
        print(e1.get())
        retailStories = makeRetailStoriesList()
        print(retailStories)
        V,W = takeVelocityAndIntervalArray()
        print(V,W)
        return simulated_annealing(1000, 225.84, 0.01, retailStories, float(c1.get()), float(a1.get()),float(d1.get()),V, W,float(b1.get()),float(e1.get()))

    calcbtn = Button(master=root, text="Calculate solution", command=calc)
    calcbtn.grid(row=1, column=0, padx="5", pady="5", sticky="w")

    root.title("My App")

    root.mainloop()