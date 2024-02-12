import os
import scipy.io
import pickle
import consolemenu
from consolemenu.items import FunctionItem
import pandas as pd
import numpy as np

output_folder = "Output"
np.set_printoptions(threshold=np.inf)

def mat_to_csv_or_pickle():
    file_path = input("Please enter the relative path to the .mat file: ")

    # Extract the file name from the path (without the .mat extension)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    output_csv_path = f"{output_folder}/{file_name}/csv"
    output_pickle_path = f"{output_folder}/{file_name}/pickle"

    try:
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)

        # Make sure that the output folders exist
        os.makedirs(output_csv_path, exist_ok=True)
        os.makedirs(output_pickle_path, exist_ok=True)

        # Display menu to choose file format
        format_menu_labels = ["Save as .csv Files", "Save as .pkl Files"]
        format_functions = [save_as_csv, save_as_pickle]
        format_menu = consolemenu.SelectionMenu(format_menu_labels, "Choose the output format", clear_screen=False)
        format_menu.show()
        format_selection = format_menu.selected_option

        if format_selection is not None:
            format_functions[format_selection](mat_data, output_csv_path, output_pickle_path)

        print("MAT file successfully processed!")

    except Exception as e:
        print(f"Error reading .mat file: {e}")

def save_as_csv(mat_data, output_csv_path, output_pickle_path):
    # Go through the variables in the .mat file data
    for variable_name in mat_data:
        variable = mat_data[variable_name]

        # Save each variable as a separate CSV file
        if isinstance(variable, (list, tuple, np.ndarray)):
            csv_file_path = f"{output_csv_path}/{variable_name}.csv"
            np.savetxt(csv_file_path, variable, delimiter=',', fmt='%s')

    print("CSV files saved in the 'Output' folder!")

def save_as_pickle(mat_data, output_csv_path, output_pickle_path):
    # Go through the variables in the .mat file data
    for variable_name in mat_data:
        variable = mat_data[variable_name]

        # Save each variable as a separate pickle file
        variable_pickle_path = f"{output_pickle_path}/{variable_name}.pkl"
        with open(variable_pickle_path, 'wb') as pickle_file:
            pickle.dump(variable, pickle_file)

    print("Pickle files saved in the 'Output' folder!")

def help_message():
    print("Save .mat File as .csv or .pkl Files - Convert a .mat file to .csv or .pkl files and save them in the 'Output' folder.")
    print("Help - Display this help message again.")
    print("Exit - Exit the program.")
    print("Note: The .csv files can be nicely viewed in Excel or any other spreadsheet software.")

def main():
    labels = ["Save .mat File as .csv or .pkl Files", "Help"]
    functions = [mat_to_csv_or_pickle, help_message]

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