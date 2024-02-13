from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import gensim.downloader
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import words

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = gensim.downloader.load('glove-wiki-gigaword-300')
nltk.download('words')
english_words = set(words.words())

# Filter the model's vocabulary
filtered_vocab = {
    word: model[word] for word in model.key_to_index
    if word in english_words
}
new_kv = KeyedVectors(vector_size=model.vector_size)

# Prepare lists of keys (words) and their vectors
keys = list(filtered_vocab.keys())
vectors = [filtered_vocab[word] for word in keys]
new_kv.sort_by_descending_frequency()
# Add all vectors in one batch
new_kv.add_vectors(keys, vectors)

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    word1 = data['word1']
    word2 = data['word2']
    similarity_score = new_kv.similarity(word1, word2)
    # Convert numpy.float32 to Python float
    similarity_score = float(similarity_score)
    return jsonify({'similarity': similarity_score})

@app.route('/difference', methods=['POST'])
def difference():
    data = request.get_json()
    word1 = data['word1']
    word2 = data['word2']
    try:
        # Find words most similar to word2 when word1 is "subtracted" from it
        # This is akin to finding what makes word2 different from word1
        results = new_kv.most_similar(positive=[word2], negative=[word1], topn=1)
        
        difference_word = results[0][0]  # The word that represents the difference
        difference_score = float(results[0][1])  # The similarity score of the difference word
        return jsonify({'difference': difference_word, 'score': difference_score})
    except Exception as e:
        print(f"Error processing difference request: {e}")
        return jsonify({"error": "Error processing request, make sure the words exist in the model"}), 500

@app.route('/test', methods=['GET'])
def test():
    result = new_kv.most_similar_cosmul(positive=['queen'], negative=['king'])
    most_similar_key, similarity = result[0]  # look at the first match
    print(f"{most_similar_key}: {similarity:.4f}")
    return 'Test endpoint is working'



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1237)
