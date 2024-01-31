import os
import scipy.io
import pickle
import consolemenu
from consolemenu.items import FunctionItem
import pandas as pd
import numpy as np

output_folder = "PickledOutput"
np.set_printoptions(threshold=np.inf)

def read_and_save_mat():
    file_path = input("Please enter the relative path to the .mat file: ")

    # Extract the file name from the path (without the .mat extension)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    output_pickle_path = f"{output_folder}/{file_name}"

    try:
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)

        # Make sure that the output folder exists
        os.makedirs(output_pickle_path, exist_ok=True)

        # Go through the variables in the .mat file data
        for variable_name in mat_data:
            variable = mat_data[variable_name]

            # Save each variable as a separate pickle file
            variable_pickle_path = f"{output_pickle_path}/{variable_name}.pkl"
            with open(variable_pickle_path, 'wb') as pickle_file:
                pickle.dump(variable, pickle_file)

        print("MAT file successfully processed and pickle files saved!")

    except Exception as e:
        print(f"Error reading .mat file: {e}")

def view_all_pickle_file_names():
    pickle_files = []

    # Go through all the files in the Output folder and add the .pkl files to the list
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            if file.endswith(".pkl"):
                pickle_files.append(os.path.join(root, file))

    for file in pickle_files:
        print(file)

def view_pickle_file():
    try:
        file_path = input("Please enter the relative path to the pickle file (with .pkl extension): ")
        if os.path.exists(file_path):
            variable_data = pd.read_pickle(file_path)
            print(f"\n{file_path}:")
            print(variable_data)

        else:
            print(f"File '{file_path}' not found.")

    except Exception as e:
        print(f"Error viewing pickle file: {e}")

def help_message():
    print("Save .mat File as a .pkl File - Convert a .mat file to .pkl files and save them in the 'PickledOutput' folder.")
    print("View all Pickle File Names - View the names of all the .pkl files in the 'PickledOutput' folder.")
    print("View Pickle File in Terminal - View the contents of a .pkl file in the terminal.")
    print("Help - Display this help message :)")
    print("Exit - Exit the program.")

def main():
    labels = ["Save .mat File as a .pkl File", "View all Pickle File Names", "View Pickle File in Terminal", "Help"]
    functions = [read_and_save_mat, view_all_pickle_file_names, view_pickle_file, help_message]

    menu = consolemenu.SelectionMenu(labels, "Please select an option", clear_screen=False)

    while True:
        menu.show()
        selection = menu.selected_option
        if selection == len(labels):
            exit()
        elif selection is not None:
            functions[selection]()

if __name__ == "__main__":
    main()
