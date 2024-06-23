import csv
import pickle
from pathlib import Path
import logging
import numpy as np
from collections import Counter
import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

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
        name = filepath.parent.name
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

def test(model: str = "hog"):
    logging.info(f"Testing model: {model}")
    test_results = []
    for filepath in Path("testing").rglob("*"):
        if filepath.is_file():
            recognized_faces = recognize_faces(image_location=str(filepath.absolute()), model=model)
            test_results.append({"image": filepath.name, "faces": recognized_faces})

    return test_results

def store_results(test_results, filename="test_results.csv"):
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Name", "ID"])
        for result in test_results:
            for face in result["faces"]:
                writer.writerow([result["image"], face["Name"], face["ID"]])
