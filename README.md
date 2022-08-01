# sentiment
pip install tensorflow scikit-learn nltk pandas matplotlib PyYAML

# Preparing data: the_edge_corpus.py, read the main() to understand the steps to prepare data

# Edit config file
nano config.yaml

# Proposed model
python lstm_att_classifier.py -c config.yaml -s all

# Baseline machine learnings: decision tree, Naive Bayes, SVM
python classification.py

# Baseline neural network models
# LSTM
python simple_lstm_classifier.py -c config.yaml -s all

# Bi-directional
python simple_bilstm.py -c config.yaml -s all

# Bi-encoders
python lstm_bi_encoders_classifier.py -c config.yaml -s all
