import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_cosine_similarity(text1, text2):
    """
    Calculate cosine similarity between two texts using Sentence-BERT embeddings.
    
    Args:
        text1 (str): The first text.
        text2 (str): The second text.
    
    Returns:
        float: Cosine similarity between the embeddings of the two texts.
    """
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]

    if np.all(embeddings1 == 0) or np.all(embeddings2 == 0):
        print(f"Warning: One of the embeddings is a zero vector for texts: \n{text1} \n{text2}")
        return 1e-10

    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    
    if np.isnan(similarity):
        print(f"Warning: Cosine Similarity is NaN for texts: \n{text1} \n{text2}")
        similarity = 1e-10
    
    return similarity

def calculate_bleu_score(reference, candidate):
    """
    Calculate the BLEU score between reference (original review) and candidate (meta-review).
    This version uses smoothing to avoid zero BLEU scores for higher-order n-grams.
    """
    reference_tokens = [word_tokenize(ref.lower()) for ref in reference]
    candidate_tokens = word_tokenize(candidate.lower())
    
    smoothing_function = SmoothingFunction().method4 
    
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
    return bleu_score

def calculate_meta_review_length_ratio(data_list):
    """
    Calculate the length ratio of each meta-review to the longest meta-review.
    
    Args:
        data_list (list): List of dictionaries containing reviews and meta-reviews.
    
    Returns:
        list: Updated data_list with added 'Length Ratio' field for each meta-review.
    """
    max_meta_review_length = max([len(word_tokenize(item['Metareview'])) for item in data_list])
    
    for item in data_list:
        meta_review_length = len(word_tokenize(item['Metareview']))
        item['Length Ratio'] = meta_review_length / max_meta_review_length
    
    return data_list

def process_reviews_and_calculate_metrics(data_list):
    """
    Process the reviews and calculate Cosine Similarity, BLEU score, and Length Ratio.
    
    Args:
        data_list (list): List of dictionaries containing reviews and meta-reviews.
    
    Returns:
        list: A list of dictionaries with the calculated metrics.
    """
    data_list = calculate_meta_review_length_ratio(data_list)
    results = []
    
    for item in data_list:
        meta_review = item['Metareview']
        reviews = item['Review']
        
        cosine_similarities = [calculate_cosine_similarity(meta_review, review) for review in reviews]
        bleu_scores = [calculate_bleu_score([review], meta_review) for review in reviews]

        
        results.append({
            'Metareview': meta_review,
            'Reviews': reviews,
            'Cosine Similarities': cosine_similarities,
            'BLEU Scores': bleu_scores,
            'Length Ratio': item['Length Ratio']
        })
    
    return results

def perform_clustering(results):
    """
    Perform K-means clustering on the calculated features (Cosine Similarity, BLEU Score, Length Ratio).
    
    Args:
        results (list): List of dictionaries containing calculated metrics for each meta-review.
    
    Returns:
        tuple: Updated results with cluster labels, K-means model, and cluster labels.
    """
    features = []
    
    for item in results:
        avg_cosine_similarity = np.mean(item['Cosine Similarities'])
        avg_bleu_score = np.mean(item['BLEU Scores'])
        if np.isnan(avg_cosine_similarity):
            print(item["Metareview"],item["Reviews"],item['Cosine Similarities'])
            raise ValueError(f"NaN value detected in meta-review with cos")
        if np.isnan(avg_bleu_score):
            raise ValueError(f"NaN value detected in meta-review with bleu")
            
        feature = [
            float(avg_cosine_similarity), 
            float(avg_bleu_score),        
            float(item['Length Ratio'])  
        ]
        
        features.append(feature)

    #print(features)
    
    features = np.array(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(features_scaled)
    
    labels = kmeans.labels_
    
    for idx, item in enumerate(results):
        item['Cluster Label'] = labels[idx]
    
    return results, labels, kmeans

def visualize_clusters(results, labels):
    """
    Visualize the clusters in 2D using PCA for dimensionality reduction.
    
    Args:
        results (list): List of dictionaries containing calculated metrics for each meta-review.
        labels (array): The cluster labels assigned by K-means clustering.
    """
    features = []
    
    for item in results:
        avg_cosine_similarity = np.mean(item['Cosine Similarities'])
        avg_bleu_score = np.mean(item['BLEU Scores'])
        
        feature = [
            float(avg_cosine_similarity), 
            float(avg_bleu_score),        
            float(item['Length Ratio'])  
        ]
        
        features.append(feature)
    
    features = np.array(features)
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis')
    plt.title("K-means Clustering (2 Clusters)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster Label")
    plt.show()

def save_results(results):
    """
    Save the clustering results to a file.
    
    Args:
        results (list): The list of results with cluster labels.
    """
    with open("separate_output.txt", "w") as f:
        for idx, item in enumerate(results):
            f.write(f"Meta-review {idx+1}{item['Metareview']}: Cluster {item['Cluster Label']}\n")