import tkinter as tk
from tkinter import filedialog, messagebox
import final_ifp2

global img_lst,mdl,final
mdl = "best_yolo.pt"
img_lst = []
final = []
def open_file_dialog():
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    img_lst.append(file_path)

    root.destroy()



root = tk.Tk()
root.title("date detector")
root.geometry("600x400")

label = tk.Label(root, text="Open image to detect date")
label.pack(pady=10)

button = tk.Button(root, text="Open File", command=open_file_dialog)
button.pack(pady=10)

root.mainloop()
root.quit()

def execute():


    final = final_ifp2.date_finder(final_ifp2.detect_multiple(img_lst, mdl))
   # final_ifp2.display_bbox(final,coord,img_lst[0])
    messagebox.showinfo("The Output", f" dates: {final}")
    #if len(final)==2:
    #    messagebox.showinfo("The Output", f"manufaccturing date: {final[0]}\nexpiry date:{final[1]}")
    #elif len(final)==1:
    #    messagebox.showinfo("The Output", f"manufaccturing date: {final[0]}")
    #else:
    #    messagebox.showinfo("The Output", "no detections...")
    root2.destroy()

root2 = tk.Tk()
root2.title("Start")
root2.geometry("600x400")

label = tk.Label(root2, text="can we start?")
label.pack(pady=10)

button = tk.Button(root2, text="start", command=execute)
button.pack(pady=10)

root2.mainloop()
root2.quit()