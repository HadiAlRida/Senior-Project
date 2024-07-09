import argparse
import logging
from pathlib import Path
import time

from recognition_testing_module import encode_known_faces, test, validate, store_results, evaluate
from network_module import build_network, detect_communities_louvain, detect_communities_girvan_newman, calculate_centrality_measures, visualize_network, export_friendship_data_to_db

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize faces in images and perform community detection")
    parser.add_argument("--train", action="store_true", help="Train on input data")
    parser.add_argument("--networking", action="store_true", help="Test the model with images in the networking folder")
    parser.add_argument("--validate", action="store_true", help="Validate the model with images in the validation folder")
    parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn", "resnet", "mtcnn"], help="Which model to use for training: hog (CPU), cnn (GPU), resnet, mtcnn")
    parser.add_argument("--visualization", choices=["louvain", "girvan-newman", "none"], default="none", help="Type of network visualization to perform (default: none)")
    parser.add_argument("--schema", default="Students", help="Database schema name")
    parser.add_argument("--table", default="friendship_network", help="Database table name")
    parser.add_argument("--detailed_table", default="detailed_friendship_data", help="Database table name for detailed friendship data")
    args = parser.parse_args()

    start_time = time.time()

    if args.train:
        train_start_time = time.time()
        encode_known_faces(model=args.m)
        train_elapsed_time = time.time() - train_start_time
        logging.info(f"Encoding known faces completed in {train_elapsed_time:.2f} seconds.")

    if args.validate:
        validate_start_time = time.time()
        true_labels, predicted_labels = validate(model=args.m)
        report, accuracy = evaluate(true_labels, predicted_labels)
        validate_elapsed_time = time.time() - validate_start_time
        logging.info(f"Validation completed in {validate_elapsed_time:.2f} seconds.\nAccuracy: {accuracy}\nReport:\n{report}")

    if args.networking:
        networking_start_time = time.time()
        test_results, co_occurrence_counter, true_labels, predicted_labels = test(model=args.m)
        store_results(test_results)
        networking_elapsed_time = time.time() - networking_start_time
        logging.info(f"Networking completed in {networking_elapsed_time:.2f} seconds. Results stored in test_results.csv.")
        
        evaluate_start_time = time.time()
        report, accuracy = evaluate(true_labels, predicted_labels)
        evaluate_elapsed_time = time.time() - evaluate_start_time
        logging.info(f"Networking Evaluation completed in {evaluate_elapsed_time:.2f} seconds.\nAccuracy: {accuracy}\nReport:\n{report}")

        network_build_start_time = time.time()
        G = build_network(test_results)
        network_build_elapsed_time = time.time() - network_build_start_time
        logging.info(f"Network built in {network_build_elapsed_time:.2f} seconds with {len(G.nodes)} nodes and {len(G.edges)} edges.")
        
        community_detection_louvain_start_time = time.time()
        partition_louvain = detect_communities_louvain(G)
        community_detection_louvain_elapsed_time = time.time() - community_detection_louvain_start_time
        logging.info(f"Detected {len(set(partition_louvain.values()))} communities using Louvain method in {community_detection_louvain_elapsed_time:.2f} seconds.")
        
        community_detection_girvan_newman_start_time = time.time()
        partition_girvan_newman = detect_communities_girvan_newman(G)
        community_detection_girvan_newman_elapsed_time = time.time() - community_detection_girvan_newman_start_time
        logging.info(f"Detected {len(partition_girvan_newman)} communities using Girvan-Newman method in {community_detection_girvan_newman_elapsed_time:.2f} seconds.")
        
        centrality_start_time = time.time()
        centrality_measures = calculate_centrality_measures(G)
        centrality_elapsed_time = time.time() - centrality_start_time
        logging.info(f"Centrality measures calculated in {centrality_elapsed_time:.2f} seconds.")
        
        if args.visualization == "louvain":
            visualization_start_time = time.time()
            logging.info("Visualizing network with Louvain partition...")
            visualize_network(G, partition_louvain, centrality_measures)
            visualization_elapsed_time = time.time() - visualization_start_time
            logging.info(f"Network visualization with Louvain partition completed in {visualization_elapsed_time:.2f} seconds.")
        elif args.visualization == "girvan-newman":
            visualization_start_time = time.time()
            logging.info("Visualizing network with Girvan-Newman partition...")
            visualize_network(G, partition_girvan_newman, centrality_measures)
            visualization_elapsed_time = time.time() - visualization_start_time
            logging.info(f"Network visualization with Girvan-Newman partition completed in {visualization_elapsed_time:.2f} seconds.")
        else:
            logging.warning("No valid visualization option selected.")
    # Export friendship data to database
        if args.networking:
            export_to_db_start_time = time.time()
            export_friendship_data_to_db(G, args.schema, args.table, args.detailed_table, partition_louvain, centrality_measures)
            export_to_db_elapsed_time = time.time() - export_to_db_start_time
            logging.info(f"Friendship data exported to database in {export_to_db_elapsed_time:.2f} seconds.")

    total_elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {total_elapsed_time:.2f} seconds.")
