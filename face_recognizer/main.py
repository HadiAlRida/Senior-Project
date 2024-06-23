import argparse
import logging
from pathlib import Path

# Importing the functions from the other modules
from recognition_testing_module import encode_known_faces, test, validate, store_results, evaluate
from network_module import build_network, detect_communities_louvain, detect_communities_girvan_newman, calculate_centrality_measures, visualize_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize faces in images and perform community detection")
    parser.add_argument("--train", action="store_true", help="Train on input data")
    parser.add_argument("--test", action="store_true", help="Test the model with images in the testing folder")
    parser.add_argument("--validate", action="store_true", help="Validate the model with images in the validation folder")
    parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn", "resnet", "mtcnn"], help="Which model to use for training: hog (CPU), cnn (GPU), resnet, mtcnn")
    args = parser.parse_args()

    if args.train:
        encode_known_faces(model=args.m)
        logging.info("Encoding known faces completed.")

    if args.validate:
        true_labels, predicted_labels = validate(model=args.m)
        report, accuracy = evaluate(true_labels, predicted_labels)
        logging.info(f"Validation completed.\nAccuracy: {accuracy}\nReport:\n{report}")

    if args.test:
        test_results, co_occurrence_counter, true_labels, predicted_labels = test(model=args.m)
        store_results(test_results)
        logging.info("Testing completed. Results stored in test_results.csv.")
        
        # Evaluate the test results
        report, accuracy = evaluate(true_labels, predicted_labels)
        logging.info(f"Testing Evaluation:\nAccuracy: {accuracy}\nReport:\n{report}")

        G = build_network(test_results)
        logging.info(f"Network built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
        
        # Perform community detection using different algorithms
        partition_louvain = detect_communities_louvain(G)
        logging.info(f"Detected {len(set(partition_louvain.values()))} communities using Louvain method.")
        
        partition_girvan_newman = detect_communities_girvan_newman(G)
        logging.info(f"Detected {len(partition_girvan_newman)} communities using Girvan-Newman method.")
        
        # Calculate centrality measures
        centrality_measures = calculate_centrality_measures(G)
        logging.info("Centrality measures calculated.")
        
        # Visualize the network with Louvain partition
        logging.info("Visualizing network with Louvain partition...")
        visualize_network(G, partition_louvain, centrality_measures)
