from the_edge_corpus import TheEdgeCorpusUtils
from multisource_classifier_utils import MultiSourceClassifierUtils
import datetime
from datetime import date
import calendar

util = TheEdgeCorpusUtils()

#util.save_info('../data/theEdge_2017-2018/', 'edge_raw_data.csv')
#util.save_info('../data/theEdge_2014-2018/', 'edge_raw.csv')
#util.annotate_company('edge_raw_data.csv', 'stock_listed.csv', 'edge_labeled.csv')

#util.setup_data('edge_labeled.csv', 'stock_listed.csv', '../Trading/data/KLSE', 20120103, 20210319, 'edge_complete.csv')
#util.prepare_data('edge_complete.csv', 'edge-text.txt', 'edge-company.txt', 'edge-label.txt', 'stock_listed.csv')

m_util = MultiSourceClassifierUtils()
#m_util.tokenize_text('data/train.unnorm.txt', 'data/train.token.txt', 'english')
#m_util.tokenize_text('data/test.unnorm.txt', 'data/test.token.txt', 'english')
#m_util.tokenize_text('data/valid.unnorm.txt', 'data/valid.token.txt', 'english')

#util.convert_to_unk('data/train.token.txt', 'data/edgeMarket_all.vocab', 'data/train.txt')
#util.convert_to_unk('data/test.token.txt', 'data/edgeMarket_all.vocab', 'data/test.txt')
#util.convert_to_unk('data/valid.token.txt', 'data/edgeMarket_all.vocab', 'data/valid.txt')