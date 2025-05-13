import utils
from data.load_data import load_data_from_json
from data.metareview_separate import *


if __name__ == "__main__":
    args = utils.get_args()
    data_list = load_data_from_json(args.data_path, args.data_option, args.key_option)
    results = process_reviews_and_calculate_metrics(data_list)
    results, labels, kmeans = perform_clustering(results)
    save_results(results)
    visualize_clusters(results, labels)