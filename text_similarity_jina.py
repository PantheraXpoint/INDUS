from embeddings.JinaCLIP import JinaCLIP
import numpy as np


def compute_text_similarity(text1, text2, model=None):
    """
    Compute cosine similarity score between two text strings using JinaCLIP model.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        model (JinaCLIP, optional): Pre-initialized JinaCLIP model. If None, creates a new one.
        
    Returns:
        float: Similarity score between -1 and 1 (typically between 0 and 1 for normalized embeddings)
    """
    # Initialize model if not provided
    if model is None:
        model = JinaCLIP()
    
    # Get embeddings for both texts
    embeddings = model.get_text_features([text1, text2])
    
    # Compute cosine similarity (dot product since embeddings are normalized)
    similarity = np.dot(embeddings[0], embeddings[1])
    
    return float(similarity)


if __name__ == "__main__":
    # Example usage
    print("Initializing JinaCLIP model...")
    jina_model = JinaCLIP()
    
    # Example 1: Similar texts
    text1 = "The red car is on the left of the white truck."
    text2 = "The red car is on the right of the white truck."
    similarity1 = compute_text_similarity(text1, text2, jina_model)
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity score: {similarity1:.4f}")
    
    # # Example 2: Different texts
    # text3 = "The weather is nice today"
    # text4 = "I love programming in Python"
    # similarity2 = compute_text_similarity(text3, text4, jina_model)
    # print(f"\nText 1: '{text3}'")
    # print(f"Text 2: '{text4}'")
    # print(f"Similarity score: {similarity2:.4f}")
    
    # # Example 3: Interactive mode
    # print("\n" + "="*50)
    # print("Interactive mode - Enter two texts to compare:")
    # print("="*50)
    # user_text1 = input("Enter first text: ")
    # user_text2 = input("Enter second text: ")
    # user_similarity = compute_text_similarity(user_text1, user_text2, jina_model)
    # print(f"\nSimilarity score: {user_similarity:.4f}")

