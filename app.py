from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from gensim.models import KeyedVectors

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Word2Vec model (this may take a few minutes)
model_path = 'GoogleNews-vectors-negative300.bin'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    word1 = data['word1']
    word2 = data['word2']
    similarity_score = word_vectors.similarity(word1, word2)
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
        results = word_vectors.most_similar(positive=[word2], negative=[word1], topn=1)
        
        difference_word = results[0][0]  # The word that represents the difference
        difference_score = float(results[0][1])  # The similarity score of the difference word
        return jsonify({'difference': difference_word, 'score': difference_score})
    except Exception as e:
        print(f"Error processing difference request: {e}")
        return jsonify({"error": "Error processing request, make sure the words exist in the model"}), 500

@app.route('/test', methods=['GET'])
def test():
    result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    most_similar_key, similarity = result[0]  # look at the first match
    print(f"{most_similar_key}: {similarity:.4f}")
    return 'Test endpoint is working'



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1237)
