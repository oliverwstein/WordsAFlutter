import random
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import gensim.downloader
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import words, wordnet
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
english_words = set(words.words())
model = gensim.downloader.load('glove-wiki-gigaword-300')

def is_valid_word(word, english_words):
    """
    Check if a word is a valid Scrabble word by including nouns, adjectives, and verbs,
    and using WordNet to attempt to exclude proper nouns.
    This function returns True if the word has at least one synset in these categories
    and is not a proper noun.
    """
    # Check if the word has synsets (i.e., is recognized by WordNet)
    synsets = wordnet.synsets(word)
    if not synsets:
        return False  # Word is not recognized by WordNet
    if word not in english_words:
        return False  # Word is not in the list of English words
    
    valid_categories = ['noun', 'adj', 'verb', 'adv']
    for synset in synsets:
        lexname = synset.lexname()
        if any(category in lexname for category in valid_categories):
            # If including proper nouns as valid, remove or adjust this check
            if 'noun' in lexname and not any(tag in lexname for tag in ['noun.person', 'noun.organization', 'noun.place']):
                return True  # Word is a noun and not identified as a proper noun
            elif 'noun' not in lexname:
                return True  # Word is an adjective or verb
    return False

# Filter the model's vocabulary
filtered_vocab = {
    word: model[word] for word in model.key_to_index
    if is_valid_word(word, english_words) and len(word) > 2
}
new_kv = KeyedVectors(vector_size=model.vector_size)

# Prepare lists of keys (words) and their vectors
keys = list(filtered_vocab.keys())
vectors = [filtered_vocab[word] for word in keys]
new_kv.sort_by_descending_frequency()
# Add all vectors in one batch
new_kv.add_vectors(keys, vectors)
print("wordCount", len(keys))
print(keys[0:10])

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    word1 = str(data['word1']).lower()
    word2 = str(data['word2']).lower()
    similarity_score = new_kv.similarity(word1, word2)
    # Convert numpy.float32 to Python float
    similarity_score = float(np.round(similarity_score, 2))
    return jsonify({'similarity': similarity_score})

@app.route('/difference', methods=['POST'])
def difference():
    data = request.get_json()
    word1 = str(data['word1']).lower()
    word2 = str(data['word2']).lower()
    try:
        results = new_kv.most_similar(positive=word1, negative = word2, restrict_vocab=50000)
        difference_word = results[0][0]  # The word that represents the difference
        difference_score = np.round(results[0][1], 2)  # The similarity score of the difference word
        return jsonify({'difference': difference_word, 'score': difference_score})
    except Exception as e:
        print(f"Error processing difference request: {e}")
        return jsonify({"error": "Error processing request, make sure the words exist in the model"}), 500
    
@app.route('/differences', methods=['POST'])
def differences():
    data = request.get_json()
    word1 = data['word1']
    word2 = data['word2']
    try:
        results = new_kv.most_similar(positive=word1, negative = word2, restrict_vocab=50000)
        top10Words = [results[i][0] for i in range(10)]
        return jsonify({'results': top10Words})
    except Exception as e:
        print(f"Error processing difference request: {e}")
        return jsonify({"error": "Error processing request, make sure the words exist in the model"}), 500

@app.route('/random_word', methods=['GET'])
def random_word():
    random_word = random.choice(list(new_kv.key_to_index.keys())[100:2000])
    return jsonify({'random_word': random_word})

@app.route('/hints', methods=['POST'])
def hints():
    data = request.get_json()
    word = str(data['word']).lower()

    # Ensure the word exists in the model to avoid errors
    if word not in new_kv.key_to_index:
        return jsonify({'error': 'Word not found in the vocabulary'}), 404

    # Calculate cosine similarities for the word against all words in the filtered model
    cosine_similarities = new_kv.cosine_similarities(new_kv.get_vector(word), new_kv.vectors)
    sorted_indices = np.argsort(-cosine_similarities)
    
    # Select indices at exponential intervals to cover a broad range of similarities
    percentile_indices = [int(2**i) for i in np.array(list(range(0, 10)))/2 if 2**i < len(sorted_indices)]
    
    percentile_words = [word]  # Start with the input word
    chosen_word_indices = [new_kv.key_to_index[word]]
    
    for a in range(len(percentile_indices) - 1):
        binIndices = sorted_indices[percentile_indices[a]:percentile_indices[a+1]]
        binWords = [new_kv.index_to_key[i] for i in binIndices if i not in chosen_word_indices]
        if binWords:
            # Calculate the word most dissimilar to those already chosen
            dissimilar_word = new_kv.most_similar_to_given(percentile_words[-1], binWords)
            percentile_words.append(dissimilar_word)
            chosen_word_indices.append(new_kv.key_to_index[dissimilar_word])
    percentile_words.reverse()
    return jsonify({'hints': percentile_words})

@app.route('/test', methods=['GET'])
def test():
    result = new_kv.most_similar_cosmul(positive=['queen'], negative=['king'])
    most_similar_key, similarity = result[0]  # look at the first match
    print(f"{most_similar_key}: {similarity:.4f}")
    return 'Test endpoint is working'



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1237)
