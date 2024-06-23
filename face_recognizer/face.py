import argparse
import pickle
import csv
import uuid
import logging
from collections import Counter
from pathlib import Path

import face_recognition
import dlib
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw
import numpy as np

logging.basicConfig(level=logging.INFO)

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
VALIDATION_RESULTS_PATH = Path("output/validation_results.csv")
UNIQUE_RESULTS_PATH = Path("output/result.csv")

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)
Path("testing").mkdir(exist_ok=True)  # Ensure the testing directory exists

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--test", action="store_true", help="Test the model with images in the testing folder")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn", "resnet", "mtcnn"], help="Which model to use for training: hog (CPU), cnn (GPU), resnet, mtcnn")
args = parser.parse_args()

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    logging.info(f"Encoding known faces using model: {model}")
    if encodings_location.exists():
        with encodings_location.open(mode="rb") as f:
            existing_encodings = pickle.load(f)
            names = existing_encodings["names"]
            encodings = existing_encodings["encodings"]
    else:
        names = []
        encodings = []

    for filepath in Path("training").rglob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = get_face_locations(image, model)
        if len(face_locations) != 1:
            logging.warning(f"Skipping {filepath}, found {len(face_locations)} faces.")
            continue

        face_encodings = face_recognition.face_encodings(image, face_locations)
        for encoding in face_encodings:
            if not any(np.allclose(encoding, existing_encoding) for existing_encoding in encodings):
                names.append(name)
                encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def get_face_locations(image, model):
    if model in ["hog", "cnn"]:
        return face_recognition.face_locations(image, model=model)
    elif model == "resnet":
        detector = dlib.get_frontal_face_detector()
        dets = detector(image, 1)
        return [(d.top(), d.right(), d.bottom(), d.left()) for d in dets]
    elif model == "mtcnn":
        detector = MTCNN()
        results = detector.detect_faces(image)
        return [(result['box'][1], result['box'][0] + result['box'][2], result['box'][1] + result['box'][3], result['box'][0]) for result in results]

def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    logging.info(f"Recognizing faces in image: {image_location} using model: {model}")
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = get_face_locations(input_image, model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    recognized_faces = []

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "unknown"
        _display_face(draw, bounding_box, name)
        recognized_faces.append({"Filename": Path(image_location).name, "Name": name, "ID": bounding_box})

    del draw
    pillow_image.show()
    
    return recognized_faces

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding, tolerance=0.6)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    
    if votes:
        return votes.most_common(1)[0][0]
    return None

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((text_left, text_top), name, fill=TEXT_COLOR)

def validate(model: str = "hog"):
    logging.info(f"Validating model: {model}")
    results = []
    unique_results = {}
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognized_faces = recognize_faces(image_location=str(filepath.absolute()), model=model)
            for face in recognized_faces:
                results.append(face)
                if face["Name"] not in unique_results:
                    unique_results[face["Name"]] = str(uuid.uuid4())

    with VALIDATION_RESULTS_PATH.open(mode="w", newline='') as csvfile:
        fieldnames = ["Filename", "Name", "ID"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    with UNIQUE_RESULTS_PATH.open(mode="w", newline='') as csvfile:
        fieldnames = ["Name", "ID"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, id in unique_results.items():
            writer.writerow({"Name": name, "ID": id})

def test(model: str = "hog"):
    logging.info(f"Testing model: {model}")
    test_results = []
    for filepath in Path("testing").rglob("*"):
        if filepath.is_file():
            recognized_faces = recognize_faces(image_location=str(filepath.absolute()), model=model)
            for face in recognized_faces:
                test_results.append(face)
    
    for result in test_results:
        logging.info(f"File: {result['Filename']}, Name: {result['Name']}, ID: {result['ID']}")

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
        
    if args.test:
        test(model=args.m)
