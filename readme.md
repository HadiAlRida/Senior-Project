# Face Recognition and Community Detection

This project aims to recognize faces in images, build a friendship network based on co-occurrences of faces, and detect communities within that network. The project consists of multiple Python modules, each performing specific tasks, such as face encoding, face recognition, network construction, community detection, and database handling.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Validation](#validation)
  - [Networking](#networking)
  - [Visualization](#visualization)
  - [Exporting to Database](#exporting-to-database)
- [Modules](#modules)
  - [network_module.py](#network_modulepy)
  - [recognition_testing_module.py](#recognition_testing_modulepy)
  - [main.py](#mainpy)
  - [database_handler.py](#database_handlerpy)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/face-recognition-network.git
    cd face-recognition-network
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To encode known faces, run:
```bash
python main.py --train --m <model>
```
Replace `<model>` with one of the following: `hog`, `cnn`, `resnet`, `mtcnn`.

### Validation
To validate the model with images in the validation folder, run:
```bash
python main.py --validate --m <model>
```

### Networking
To test the model with images in the networking folder and perform community detection, run:
```bash
python main.py --networking --m <model> --visualization <type>
```
Replace `<type>` with `louvain`, `girvan-newman`, or `none`.

### Visualization
The `--visualization` option specifies the type of network visualization to perform. Options include `louvain` and `girvan-newman`.

### Exporting to Database
To export the friendship network data to a database, make sure to include the `--networking` option and specify the database schema and table names:
```bash
python main.py --networking --schema <schema_name> --table <table_name> --detailed_table <detailed_table_name>
```

## Modules

### network_module.py
Handles network construction, community detection, centrality measures calculation, visualization, and exporting data to a database.

Functions:
- `build_network(results)`
- `detect_communities_louvain(G)`
- `detect_communities_girvan_newman(G)`
- `detect_communities_agglomerative(G)`
- `frequent_pattern_mining(G)`
- `calculate_centrality_measures(G)`
- `visualize_network(G, partition, centrality_measures)`
- `export_friendship_data_to_db(G, schema_name, table_name, detailed_table_name, partition, centrality_measures, filename="friendship_data.csv")`

### recognition_testing_module.py
Handles face encoding, face recognition, testing, validation, and evaluation.

Functions:
- `encode_known_faces(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH)`
- `recognize_faces(image_location, model="hog", encodings_location=DEFAULT_ENCODINGS_PATH)`
- `_recognize_face(unknown_encoding, loaded_encodings)`
- `load_ground_truth(filepath)`
- `test(model="hog")`
- `store_results(test_results, filename="test_results.csv")`
- `evaluate(true_labels, predicted_labels)`
- `get_face_locations(image, model)`
- `validate(model="hog")`

### main.py
Main script for running the various components of the project. Uses argparse to handle command-line arguments.

### database_handler.py
Handles database connections, queries, and data manipulation.

Functions:
- `create_connection()`
- `return_query(db_session, query)`
- `return_data_as_df(file_executor, input_type, db_session=None)`
- `execute_query(db_session, query)`
- `return_create_statement_from_df(dataframe, schema_name, table_name)`
- `return_insert_into_sql_statement_from_df(dataframe, schema_name, table_name)`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

