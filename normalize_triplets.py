import srsly
from tqdm import tqdm


TRIPLET_FN = '../320k_triplets.jsonl'
OUT_FN = TRIPLET_FN.replace('.jsonl', '_normalized.jsonl')


if __name__ == "__main__":
    new_triplets = []
    original_triplets = srsly.read_jsonl(TRIPLET_FN)
    for idx, triplet in tqdm(enumerate(original_triplets)):
        query_id = triplet[0]
        pos = triplet[1]
        negs = triplet[2:]
        new_triplet = [query_id]

        # Extract scores from positive and negatives
        scores = [float(pos[1])] + [float(neg[1]) for neg in negs]
        
        # Perform min-max normalization within the triplet
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero

        normalized_scores = [(score - min_score) / (max_score - min_score + 1e-8) for score in scores]
        
        # Update the scores in the triplet
        new_triplet.append([pos[0], normalized_scores[0]])
        for i, neg in enumerate(negs):
            new_triplet.append([neg[0], normalized_scores[i+1]])
        
        # Assert the final line length is 33 (1 query_id + 1 positive + 31 negatives)
        assert len(new_triplet) == 17, f"Expected 17 items, got {len(new_triplet)}"
        
        # Append to new_triplets
        new_triplets.append(new_triplet)

    srsly.write_jsonl(OUT_FN, new_triplets)