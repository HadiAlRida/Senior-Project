import csv
import pickle
from pathlib import Path
import logging
import numpy as np
from collections import Counter
import face_recognition
from sklearn.metrics import classification_report, accuracy_score
import dlib
import json

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)
Path("testing").mkdir(exist_ok=True)

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    if encodings_location.exists():
        with encodings_location.open(mode="rb") as f:
            existing_encodings = pickle.load(f)
            names = existing_encodings["names"]
            encodings = existing_encodings["encodings"]
    else:
        names = []
        encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name  # Using the folder name as the label
        image = face_recognition.load_image_file(filepath)

        face_locations = get_face_locations(image, model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            if not any(np.allclose(encoding, existing_encoding) for existing_encoding in encodings):
                names.append(name)
                encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = get_face_locations(input_image, model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    recognized_faces = []

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "unknown"
        recognized_faces.append({"Filename": Path(image_location).name, "Name": name, "ID": unknown_encoding})
    
    return recognized_faces

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding, tolerance=0.5)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    
    if votes:
        return votes.most_common(1)[0][0]
    return None

def load_ground_truth(filepath: str):
    with open(filepath, 'r') as file:
        ground_truth = json.load(file)
    return ground_truth

def test(model: str = "hog"):
    logging.info(f"Testing model: {model}")
    test_results = []
    true_labels = []
    predicted_labels = []
    co_occurrence_counter = Counter()

    # Load ground truth for testing images
    ground_truth = load_ground_truth("ground_truth.json")

    for filepath in Path("testing").rglob("*"):
        if filepath.is_file():
            recognized_faces = recognize_faces(image_location=str(filepath.absolute()), model=model)
            test_results.append({"image": filepath.name, "faces": recognized_faces})

            # Get true labels from ground truth
            true_names = ground_truth.get(filepath.name, [])

            if not true_names:
                continue  # Skip images that are not in the ground truth

            # Collect true and predicted labels
            matched_true_labels = []
            for face in recognized_faces:
                predicted_label = face["Name"]
                predicted_labels.append(predicted_label)

                if predicted_label in true_names:
                    matched_true_labels.append(predicted_label)
                else:
                    matched_true_labels.append("unknown")

                # Count co-occurrences
                for i, face_i in enumerate(recognized_faces):
                    for j, face_j in enumerate(recognized_faces):
                        if i != j:
                            pair = tuple(sorted((face_i["Name"], face_j["Name"])))
                            co_occurrence_counter[pair] += 1

            # Append matched true labels to true_labels
            true_labels.extend(matched_true_labels)

    return test_results, co_occurrence_counter, true_labels, predicted_labels


def store_results(test_results, filename="test_results.csv"):
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Name", "ID"])
        for result in test_results:
            for face in result["faces"]:
                writer.writerow([result["image"], face["Name"], face["ID"]])

def evaluate(true_labels, predicted_labels):
    logging.info(f"Number of true labels: {len(true_labels)}")
    logging.info(f"Number of predicted labels: {len(predicted_labels)}")
    
    report = classification_report(true_labels, predicted_labels, zero_division=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return report, accuracy


def get_face_locations(image, model):
    if model in ["hog", "cnn"]:
        return face_recognition.face_locations(image, model=model)
    elif model == "resnet":
        detector = dlib.get_frontal_face_detector()
        dets = detector(image, 1)
        return [(d.top(), d.right(), d.bottom(), d.left()) for d in dets]
    elif model == "mtcnn":
        from mtcnn import MTCNN
        detector = MTCNN()
        results = detector.detect_faces(image)
        return [(result['box'][1], result['box'][0] + result['box'][2], result['box'][1] + result['box'][3], result['box'][0]) for result in results]
    return []

def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    true_labels = []
    predicted_labels = []
    unique_ids = []
    
    for folder in Path("validation").iterdir():
        if folder.is_dir():
            true_label = folder.name
            for filepath in folder.rglob("*"):
                if filepath.is_file():
                    recognized_faces = recognize_faces(
                        image_location=str(filepath.absolute()), model=model
                    )
                    
                    # Since these are solo images, there should be only one face per image
                    if recognized_faces:
                        predicted_label = recognized_faces[0]["Name"]
                        unique_id = recognized_faces[0]["ID"]
                    else:
                        predicted_label = "unknown"
                        unique_id = "unknown"
                    
                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)
                    unique_ids.append({"Name": predicted_label, "ID": unique_id})

    # Save the unique names and IDs to a CSV file
    with open("output/unique_ids.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "ID"])
        for entry in unique_ids:
            writer.writerow([entry["Name"], entry["ID"]])

    return true_labels, predicted_labels
