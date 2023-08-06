import csv
import random

def generate_presence_absence_csv(class_names, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(class_names)
        presence_absence = [random.choice([0, 1]) for _ in range(len(class_names))]
        writer.writerow(presence_absence)

if __name__ == "__main__":
    with open("coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    presence_absence_csv = "presence_absence.csv"
    generate_presence_absence_csv(class_names, presence_absence_csv)
    print(f"Randomly generated presence_absence.csv with {len(class_names)} classes.")

