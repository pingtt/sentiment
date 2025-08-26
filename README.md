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

# Citation
Tan, TP., Soon, P.C., Aleasa, M., Chan, H.Y., Gan, K.H. (2023). A Deep Learning Model with Name Attention to Predict the Stock Trend from News Headline. In: Kang, DK., Alfred, R., Ismail, Z.I.B.A., Baharum, A., Thiruchelvam, V. (eds) Proceedings of the 9th International Conference on Computational Science and Technology. ICCST 2022. Lecture Notes in Electrical Engineering, vol 983. Springer, Singapore. https://doi.org/10.1007/978-981-19-8406-8_29. 
