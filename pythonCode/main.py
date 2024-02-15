# import the neccesry libraries.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tkinter as tk
from tkinter import Text, font, Entry, Label
from tkinter import ttk  

# Function to preprocess data and train a model.
def train_and_evaluate_classifier(classifier, X_train, X_test, y_train, y_test, label_encoder, class_label):
    # Train the classifier.
    classifier.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = classifier.predict(X_test)

    # Determine the encoded label for the positive class.
    positive_label = label_encoder.transform([class_label])[0]

    # Evaluate metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_label, average='binary')
    recall = recall_score(y_test, y_pred, pos_label=positive_label, average='binary')

    return accuracy, precision, recall

# Read data from CSV file.
file_path = "covid_data.csv"
df = pd.read_csv(file_path)

# Convert categorical columns to numerical using Label Encoding.
label_columns = ['Gender', 'Location', 'Fever', 'Cough', 'Shortness_of_Breath', 'Fatigue', 'Loss_of_Taste_or_Smell',
                 'Diabetes', 'Hypertension', 'COVID_Test_Result']

le = LabelEncoder()
for col in label_columns:
    df[col] = le.fit_transform(df[col])

# Define features and target.
X = df.drop(['Patient_ID', 'COVID_Test_Result'], axis=1)
y = df['COVID_Test_Result']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Decision Tree classifier.
decision_tree_classifier = DecisionTreeClassifier(random_state=42)

# Create MLP Classifier (Neural Network).
ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)

# Train Decision Tree classifier.
decision_tree_classifier.fit(X_train, y_train)

# Train ANN classifier.
ann_classifier.fit(X_train_scaled, y_train)

# Function to perform analysis using Decision Tree and display results in a new window.
def analysis_decision_tree():
    dt_accuracy, dt_precision, dt_recall = train_and_evaluate_classifier(
        decision_tree_classifier, X_train, X_test, y_train, y_test, le, 'Positive'
    )

    # Create a new window.
    results_window = tk.Toplevel(root)
    results_window.title("Decision Tree Analysis Results")
    results_window.geometry("400x200")

    # Create a Text widget to display results.
    text_widget = Text(results_window, wrap="word", font=font.Font(family="Times New Roman", size=12))

    # Insert results into the Text widget
    text_widget.insert(tk.END, f"Accuracy: {dt_accuracy:.2f}\n")
    text_widget.insert(tk.END, f"Precision: {dt_precision:.2f}\n")
    text_widget.insert(tk.END, f"Recall: {dt_recall:.2f}\n")

    # Disable editing in the Text widget.
    text_widget.config(state=tk.DISABLED)

    # Pack the Text widget
    text_widget.pack()

# Function to perform analysis using ANN and display results in a new window.
def analysis_ann():
    ann_accuracy, ann_precision, ann_recall = train_and_evaluate_classifier(
        ann_classifier, X_train_scaled, X_test_scaled, y_train, y_test, le, 'Positive'
    )

    # Create a new window.
    results_window = tk.Toplevel(root)
    results_window.title("ANN Analysis Results")
    results_window.geometry("400x200")

    # Create a Text widget to display results.
    text_widget = Text(results_window, wrap="word", font=font.Font(family="Times New Roman", size=12))

    # Insert results into the Text widget
    text_widget.insert(tk.END, f"Accuracy: {ann_accuracy:.2f}\n")
    text_widget.insert(tk.END, f"Precision: {ann_precision:.2f}\n")
    text_widget.insert(tk.END, f"Recall: {ann_recall:.2f}\n")

    # Disable editing in the Text widget.
    text_widget.config(state=tk.DISABLED)

    # Pack the Text widget.
    text_widget.pack()


# Function to open a window for COVID prediction.
def open_covid_prediction_window():
    yes_no_codes = {"No": 0, "Yes": 1}
    covid_prediction_window = tk.Toplevel(root)

    covid_prediction_window.title("COVID Prediction")
    covid_prediction_window.geometry("500x500")  

    # Label and Entry for Patient ID.
    label_patient_id = Label(covid_prediction_window, text="Patient ID:")
    label_patient_id.grid(row=0, column=0, padx=10, pady=5)
    Patient_ID = Entry(covid_prediction_window)
    Patient_ID.grid(row=0, column=1, padx=10, pady=5)

    # Label and Entry for Age.
    label_age = Label(covid_prediction_window, text="Age:")
    label_age.grid(row=1, column=0, padx=10, pady=5)
    Age = Entry(covid_prediction_window)
    Age.grid(row=1, column=1, padx=10, pady=5)

    # Label and ComboBox for Gender.
    label_gender = Label(covid_prediction_window, text="Gender:")
    label_gender.grid(row=2, column=0, padx=10, pady=5)
    # Radio buttons for gender selection.
    Gender = tk.IntVar() 
    radio_male = tk.Radiobutton(covid_prediction_window, text="Male", variable=Gender, value=0)
    radio_male.grid(row=2, column=1, padx=10, pady=5)  # Adjust grid layout as neede
    radio_female = tk.Radiobutton(covid_prediction_window, text="Female", variable=Gender, value=1)
    radio_female.grid(row=2, column=2, padx=10, pady=5)  # Adjust grid layout as needed

    # Label and Entry for Location.
    label_location = Label(covid_prediction_window, text="Location:")
    label_location.grid(row=3, column=0, padx=10, pady=5)
    Location = tk.IntVar()
    radio_A = tk.Radiobutton(covid_prediction_window, text="Ramallah", variable=Location, value=0)
    radio_A.grid(row=3, column=1, padx=10, pady=5)  # Adjust grid layout as needed
    radio_B = tk.Radiobutton(covid_prediction_window, text="Al Bireh", variable=Location, value=1)
    radio_B.grid(row=3, column=2, padx=10, pady=5)  # Adjust grid layout as needed
    radio_C = tk.Radiobutton(covid_prediction_window, text="Beitunia", variable=Location, value=2)
    radio_C.grid(row=3, column=3, padx=10, pady=5)  # Adjust grid layout as needed


    # Label and ComboBox for Fever.
    label_fever = Label(covid_prediction_window, text="Fever:")
    label_fever.grid(row=4, column=0, padx=10, pady=5)
    fever_options = ['Yes', 'No']
    Fever = tk.StringVar()
    Fever.set("")
    combo_fever = ttk.Combobox(covid_prediction_window, textvariable=Fever, values=fever_options)
    combo_fever.grid(row=4, column=1, padx=10, pady=5)

    # Label and ComboBox for Cough.
    label_cough = Label(covid_prediction_window, text="Cough:")
    label_cough.grid(row=5, column=0, padx=10, pady=5)
    Cough = tk.StringVar()
    Cough.set("")
    combo_cough = ttk.Combobox(covid_prediction_window, textvariable=Cough, values=fever_options)
    combo_cough.grid(row=5, column=1, padx=10, pady=5)

    # Label and ComboBox for Shortness_of_Breath.
    label_breath = Label(covid_prediction_window, text="Shortness_of_Breath:")
    label_breath.grid(row=6, column=0, padx=10, pady=5)
    Shortness_of_Breath = tk.StringVar()
    Shortness_of_Breath.set("")
    combo_breath = ttk.Combobox(covid_prediction_window, textvariable=Shortness_of_Breath, values=fever_options)
    combo_breath.grid(row=6, column=1, padx=10, pady=5)

    # Label and ComboBox for Fatigue.
    label_fatigue = Label(covid_prediction_window, text="Fatigue:")
    label_fatigue.grid(row=7, column=0, padx=10, pady=5)
    fatigue_options = ['Yes', 'No']
    Fatigue = tk.StringVar()
    Fatigue.set("")
    combo_fatigue = ttk.Combobox(covid_prediction_window, textvariable=Fatigue, values=fatigue_options)
    combo_fatigue.grid(row=7, column=1, padx=10, pady=5)

    # Label and ComboBox for Loss_of_Taste_or_Smell.
    label_loss_of_taste = Label(covid_prediction_window, text="Loss_of_Taste_or_Smell:")
    label_loss_of_taste.grid(row=8, column=0, padx=10, pady=5)
    Loss_of_Taste_or_Smell = tk.StringVar()
    Loss_of_Taste_or_Smell.set("")
    combo_loss_of_taste = ttk.Combobox(covid_prediction_window, textvariable=Loss_of_Taste_or_Smell, values=fatigue_options)
    combo_loss_of_taste.grid(row=8, column=1, padx=10, pady=5)

    # Label and ComboBox for Diabetes.
    label_diabetes = Label(covid_prediction_window, text="Diabetes:")
    label_diabetes.grid(row=9, column=0, padx=10, pady=5)
    Diabetes = tk.StringVar()
    Diabetes.set("")
    combo_diabetes = ttk.Combobox(covid_prediction_window, textvariable=Diabetes, values=fatigue_options)
    combo_diabetes.grid(row=9, column=1, padx=10, pady=5)

    # Label and ComboBox for Hypertension.
    label_hypertension = Label(covid_prediction_window, text="Hypertension:")
    label_hypertension.grid(row=10, column=0, padx=10, pady=5)
    Hypertension = tk.StringVar()
    Hypertension.set("")
    combo_hypertension = ttk.Combobox(covid_prediction_window, textvariable=Hypertension, values=fatigue_options)
    combo_hypertension.grid(row=10, column=1, padx=10, pady=5)


    # Botton to predict COVID result.
    button_predict_covid = tk.Button(covid_prediction_window, text="Predict the COVID Result",
                                     command=lambda: predict_covid_result(
                                         Patient_ID.get(), Age.get(), Gender.get(),
                                         Location.get(), yes_no_codes[Fever.get()],
                                         yes_no_codes[Cough.get()],
                                         yes_no_codes[Fatigue.get()],
                                         yes_no_codes[combo_fatigue.get()],
                                         yes_no_codes[Loss_of_Taste_or_Smell.get()],
                                         yes_no_codes[Diabetes.get()],
                                         yes_no_codes[Hypertension.get()]
                                     ))
    button_predict_covid.grid(row=13, column=0, columnspan=2, pady=10)


def predict_covid_result(Patient_ID, Age, Gender, Location, Fever, Cough, Shortness_of_Breath, Fatigue, Loss_of_Taste_or_Smell,
                          Diabetes, Hypertension):
    # Convert categorical features to numerical using Label Encoding.
    input_data = {
        'Patient_ID': [int(Patient_ID)],
        'Age': [int(Age)],
        'Gender': [str(Gender)],
        'Location': [str(Location)],
        'Fever': [str(Fever)],
        'Cough': [str(Cough)],
        'Shortness_of_Breath': [str(Shortness_of_Breath)],
        'Fatigue': [str(Fatigue)],
        'Loss_of_Taste_or_Smell': [str(Loss_of_Taste_or_Smell)],
        'Diabetes': [str(Diabetes)],
        'Hypertension': [str(Hypertension)],
    }

    input_df = pd.DataFrame.from_dict(input_data)

    # Apply label encoding consistently to categorical features.
    for col in label_columns[:-1]:  # Exclude 'COVID_Test_Result' from label encoding
        le = LabelEncoder()
        le.fit(df[col])  # Use only the training data for fitting.
        input_df[col] = le.transform(input_df[col])

    # Make predictions using the trained Decision Tree classifier.
    dt_prediction = decision_tree_classifier.predict(input_df.drop(['Patient_ID'], axis=1))

    # Make predictions using the trained ANN classifier.
    input_scaled = scaler.transform(input_df.drop(['Patient_ID'], axis=1))
    ann_prediction = ann_classifier.predict(input_scaled)

    # Decode the predictions for display
    predicted_covid_dt = le.inverse_transform(dt_prediction)[0]
    predicted_covid_ann = le.inverse_transform(ann_prediction)[0]

    # Map numeric predictions to labels
    result_map = {0: "Negative", 1: "Positive"}

    # Display the predictions
    result_window = tk.Toplevel(root)
    result_window.title("COVID Prediction Result")
    result_window.geometry("400x150")

    text_widget = Text(result_window, wrap="word", font=font.Font(family="Times New Roman", size=12))

    text_widget.insert(tk.END, f"ANN Analysis Result: {result_map[predicted_covid_ann]}\n")
    text_widget.insert(tk.END, f"Decision Tree Analysis Result: {result_map[predicted_covid_dt]}\n")

    text_widget.config(state=tk.DISABLED)
    text_widget.pack()


# Create GUI.
root = tk.Tk()
root.title("COVID-19 Analysis")
root.geometry("400x300")

# Welcome labels.
welcome_label = tk.Label(root, text="Welcome to our project!", font=("Times New Roman", 14))
welcome_label.pack(pady=(20, 10)) 

subtitle_label = tk.Label(root, text="Learning supervised for COVID-19", font=("Times New Roman", 12))
subtitle_label.pack(pady=(0, 10))

# Button to perform Decision Tree analysis.
button_decision_tree = tk.Button(root, text="Analysis using Decision Tree technique", command=analysis_decision_tree)
button_decision_tree.pack(pady=10)

# Button to perform ANN analysis.
button_ann = tk.Button(root, text="Analysis using ANN technique", command=analysis_ann)
button_ann.pack(pady=10)

# Button for COVID prediction.
button_covid_prediction = tk.Button(root, text="COVID Prediction", command=open_covid_prediction_window)
button_covid_prediction.pack(pady=10)


# Center the window.
root.eval('tk::PlaceWindow . center')

root.mainloop()

