import os
import csv
import re
import pandas as pd
import numpy as np
import datetime
import random
from multisource_classifier_utils import MultiSourceClassifierUtils


class TheEdgeCorpusUtils:

    def save_info(self, dir, csv_file):
        '''
        Extract all news and published time from directory.
        :param dir:
        :param csv_file:
        :return:
        '''

        docs = list()

        for path in os.listdir(dir):
            file_path = os.path.join(dir, path)
            if os.path.isfile(file_path):
                title_time = self.extract_info(file_path)
                if title_time is not None:
                    info = list(title_time)
                    docs.append(info)

        df = pd.DataFrame(docs)
        df.columns = ["Title", "Time", "Content"]
        df.to_csv(csv_file, index=False, header=True)

    def extract_info(self, html_file):
        '''
        Extract title and publish time of theEdge article
        :param html:
        :return:
        '''

        publish_prefix = '<meta property="article:published_time" content="'
        publish_suffix = '" />'
        prefix_publish_length = len(publish_prefix)
        publish_suffix_length = len(publish_suffix)
        title_prefix = '<title>'
        title_suffix = '</title>'
        prefix_title_length = len(title_prefix)
        suffix_title_length = len(title_suffix)
        content_prefix = '<meta property="og:description" content="'
        prefix_content_length = len(content_prefix)
        got_publish = False
        got_title = False
        got_content = False
        content = ''

        with open(html_file, 'r') as reader:
            line = reader.readline()
            while line:
                if got_publish is False and line.startswith(publish_prefix):
                    publish_time = line[prefix_publish_length:-publish_suffix_length - 1]
                    got_publish = True
                elif got_title is False and line.startswith(title_prefix):
                    title = line[prefix_title_length:-suffix_title_length - 1]
                    got_title = True
                elif got_content is False and line.startswith(content_prefix):
                    content = line[prefix_content_length:-publish_suffix_length - 1]
                    got_content = True
                line = reader.readline()
        if got_title and got_publish:
            return title, publish_time, content
        else:
            return None

    def annotate_company(self, news_csv, stock_listed_csv, annotated_csv):
        '''
        Check a headline whether it contains the name of a company
        :param news_csv:
        :param stock_listed_csv:
        :param annotated_csv:
        :return:
        '''

        name_list1, name_list2 = self.load_company_names(stock_listed_csv)

        annotated_doc = list()
        with open(news_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers
            for row in csv_reader:
                text = row[0].lower()
                # print(text)
                match_list = self.scan_name_in_text2(text, name_list1)
                for item in match_list:
                    info = list()
                    info.append(row[0])
                    info.append(row[1])
                    label = name_list2[item[1]]
                    info.append(label + ' ')
                    annotated_doc.append(info)
                    # print(info)

        df = pd.DataFrame(annotated_doc)
        df.columns = ["Title", "Time", "Stock"]
        df.to_csv(annotated_csv, index=False, header=True)

    def load_company_names(self, stock_listed_csv):

        # Company name
        name_list1 = list()
        # Stock quote
        name_list2 = list()

        with open(stock_listed_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                name_list1.append(row[0].lower())
                name_list2.append(row[1].lower())

        return name_list1, name_list2

    def load_stock_name_map(self, stock_listed_csv):

        # Company name
        stock_name_map = dict()

        with open(stock_listed_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                stock_name_map[row[1].lower()] = row[0]

        return stock_name_map

    def scan_name_in_text(self, text: str, name_list):
        for i, name in enumerate(name_list):
            pattern = r'\b' + name + r'\b'
            if re.search(pattern, text):
                return name, i
        return None, None

    def scan_name_in_text2(self, text: str, name_list):
        match_list = list()
        for i, name in enumerate(name_list):
            pattern = r'\b' + name + r'\b'
            if re.search(pattern, text):
                match_list.append((name, i))
        return match_list

    def get_date_time(self, text: str):
        date_str = text[0: 10]
        date_yyyymmdd = date_str.split('-')
        year = date_yyyymmdd[0]
        month = date_yyyymmdd[1]
        day = date_yyyymmdd[2]
        hour = text[11: 13]

        return year, month, day, hour

    def load_stock_name2quote(self, stock_listed_csv):
        '''
        Get mapping from stock name to stock quote
        :param stock_listed_csv:
        :return:
        '''

        name2quote = dict()
        with open(stock_listed_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                name2quote[row[1]] = row[2]
        return name2quote

    def load_stock_data(self, stock_file, start_date, end_date, header=True):
        """
        Load stock data as a Dictionary object. Access the stock data using date (in integer). Generate technical
        analysis too.
        :param stock_file: csv file containing date, stock open, high, low, close, volume. e.g. CIMB= 1023.txt
        :param start_date: load the stock information from the start date
        :param end_date: load the stock information until (include) the end date
        :param header: whether there is header for the stock file
        :return: stock data in dictionary object. Access through date.
        """

        stock_data = dict()
        stock_info, dates = self.load_stock_as_list(stock_file, start_date, end_date, header)

        for i, item in enumerate(stock_info):
            arr = np.array(item[1:6])
            stock_data[stock_info[i][0]] = arr

        return stock_data, dates

    def load_stock_as_list(self, stock_file, start_date, end_date, header=True):
        """
        Similar to load_stock_data, but it load it as a list instead of a Dictionary object.
        :param stock_file:
        :param start_date:
        :param end_date:
        :param header:
        :return:
        """

        stock_info = list()
        dates = list()

        with open(stock_file, 'r') as read_obj:
            # quote = Path(stock_file).stem
            csv_reader = csv.reader(read_obj)
            if header:
                # skip the header
                header_item = next(csv_reader)
            for items in csv_reader:
                price_info = list()
                date_ddmmyyyy = items[0].split('/')
                date = int(date_ddmmyyyy[2] + date_ddmmyyyy[1] + date_ddmmyyyy[0])
                if start_date <= date <= end_date:
                    price_info.append(date)
                    for item in items[1:6]:
                        price_info.append(float(item))
                    stock_info.append(price_info)
                    dates.append(date)

        return stock_info, dates

    def read_stock_list(self, csv_file, column=2):
        """
        Read stock quotes from csv file
        :param csv_file:
        :param column: which column contains the stock quote
        :return:
        """
        stock_list = list()
        with open(csv_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for data in csv_reader:
                stock_list.append(data[column])

        return stock_list

    def load_all_stocks(self, stock_list_csv, stock_dir, start_date, end_date):

        # start_date = 20120101
        # end_date = 20211231
        # stock_dir = '../Trading/data/KLSE'
        # stock_list_csv = '../Trading/info/KLSE/train_list_all.csv'

        stock_list = self.read_stock_list(stock_list_csv)

        all_stock_data = dict()
        for quote in stock_list:
            csv_stock_file = stock_dir + '/' + quote + '.csv'
            if not os.path.isfile(csv_stock_file):
                continue

            stock_data, dates = self.load_stock_data(csv_stock_file, start_date, end_date)
            all_stock_data[quote] = (stock_data, dates)

        return all_stock_data

    def setup_data(self, annotated_csv, stock_list_csv, stock_dir, start_date, end_date, output_csv):
        """
        Prepare data for next day price trend prediction using headline
        :param annotated_csv:
        :param stock_list_csv:
        :param stock_dir:
        :param start_date:
        :param end_date:
        :param output_csv:
        :return:
        """

        docs = list()
        all_stock_data = self.load_all_stocks(stock_list_csv, stock_dir, start_date, end_date)
        name2quote = self.load_stock_name2quote(stock_list_csv)
        with open(annotated_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers
            for row in csv_reader:
                text = row[0].lower()
                publish_date = row[1]
                year, month, day, hour = self.get_date_time(publish_date)
                d = int(year + month + day)
                if d < start_date or d > end_date:
                    continue
                stock_name = row[2].strip().upper()
                quote = name2quote.get(stock_name)
                if quote is not None:
                    data = all_stock_data.get(quote)
                    if data is None:
                        continue
                    else:
                        stock_data = data[0]
                        dates = data[1]
                    price_before, price_after = self.get_stock_price(publish_date, stock_data, dates)
                    if price_before is not None:
                        info = list()
                        info.append(text)
                        info.append(publish_date)
                        info.append(stock_name)
                        info.append(quote)
                        info.append(price_before)
                        info.append(price_after)
                        docs.append(info)

        df = pd.DataFrame(docs)
        if len(docs) > 0:
            df.columns = ["Title", "Time", "Name", "Quote", "Before", "After"]
            df.to_csv(output_csv, index=False, header=True)
        else:
            print('WARNING: No data was captured!')

    def get_stock_price(self, publish_date, stock_data: dict, dates):

        year, month, day, hour = self.get_date_time(publish_date)
        p_date = datetime.date(int(year), int(month), int(day))
        is_weekday = self.is_weekday(p_date)
        date_int = int(year + month + day)
        hour = int(hour)

        if hour >= 9 and hour < 17 and is_weekday:
            # use today open price
            stock_prices = stock_data.get(date_int)
            if stock_prices is None:
                # brute force search
                index = self.brute_force_search(date_int, dates)
                if index < 0:
                    # stock data not found
                    return None, None
                else:
                    # return open price and close price
                    return stock_data[dates[index]][0], stock_data[dates[index]][3]
            else:
                # return open price and close price
                return stock_prices[0], stock_prices[3]
        else:
            index = self.brute_force_search(date_int, dates)
            if index < 0:
                return None, None
            else:
                if hour < 9:
                    # use the day before closing price, and today closing price
                    return stock_data[dates[index - 1]][3], stock_data[dates[index]][3]
                if hour < 24:
                    # use today closing price, and next day closing price
                    if len(dates) <= index + 1:
                        return None, None
                    else:
                        return stock_data[dates[index]][3], stock_data[dates[index + 1]][3]

    def is_weekday(self, date_time):
        monday_sunday = date_time.strftime("%A")
        if monday_sunday == 'Monday' or monday_sunday == 'Tuesday' or monday_sunday == 'Wednesday' or \
                monday_sunday == 'Thursday' or monday_sunday == 'Friday':
            return True
        else:
            return False

    def brute_force_search(self, date_int, dates):

        for i, d in enumerate(dates):
            if d == date_int:
                return i
            if d > date_int:
                p_date_str = str(date_int)
                p_date = datetime.date(int(p_date_str[0:4]), int(p_date_str[4:6]), int(p_date_str[6:8]))

                i_date_str = str(d)
                i_date = datetime.date(int(i_date_str[0:4]), int(i_date_str[4:6]), int(i_date_str[6:8]))

                n_day = i_date - p_date
                # only consider sentiment less than 3 days
                if n_day.days < 3:
                    return i
                else:
                    return -1
        return -1

    def prepare_data(self, complete_csv, text_file, company_file, label_file, stock_list_csv, shuffle=True,
                     split_dir='data/', train=0.7, test=0.1):

        docs = list()
        stock2name_map = self.load_stock_name_map(stock_list_csv)
        with open(complete_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers
            for row in csv_reader:
                sample = list()
                text = row[0].lower()
                company = stock2name_map[row[2].lower()].lower()
                trend = self.get_trend(float(row[4]), float(row[5]))
                sample.append(text)
                sample.append(company)
                sample.append(trend)
                docs.append(sample)

        if shuffle:
            random.shuffle(docs)

        with open(text_file, 'w') as t_writer, open(company_file, 'w') as c_writer, open(label_file, 'w') as l_writer:
            for d in docs:
                t_writer.write(d[0] + '\n')
                c_writer.write(d[1] + '\n')
                l_writer.write(str(d[2]) + '\n')

        # Split the data for training, testing and validation
        pos1 = round(len(docs) * 0.7)
        pos2 = pos1 + round(len(docs) * 0.1)
        train_list = docs[0:pos1]
        test_list = docs[pos1:pos2]
        valid_list = docs[pos2:]
        all_list = [train_list, test_list, valid_list]

        ml = ['train', 'test', 'valid']
        types = ['.unnorm.txt', '.att', '.lbl']
        for i, my_list in enumerate(all_list):
            with open(split_dir + ml[i] + types[0], 'w') as t_writer, \
                    open(split_dir +  ml[i] + types[1], 'w') as c_writer, \
                    open(split_dir + ml[i] + types[2], 'w') as l_writer:
                for d in my_list:
                    t_writer.write(d[0] + '\n')
                    c_writer.write(d[1] + '\n')
                    l_writer.write(str(d[2]) + '\n')

    def get_trend(self, price_before: float, price_after: float):
        """
        Classify changes to 3 classes
        :param price_before:
        :param price_after:
        :return:
        """

        changes = (price_after - price_before) / price_before

        if changes <= -0.01:
            # drop more than 0.5%
            return 0
        elif changes >= 0.01:
            # increase more than 0.5%
            return 2
        else:
            return 1

    def read_vocab(self, vocab_file):
        vocab_set = set()

        with open(vocab_file, 'r', encoding='utf8') as reader:
            lines = reader.readlines()
            for line in lines:
                vocab_set.add(line.rstrip())

        return vocab_set

    def convert_to_unk(self, input_file, vocab_file, output_file):

        unk = "<UNK>"
        vocab_set = self.read_vocab(vocab_file)

        with open(input_file, 'r', encoding='utf8') as reader:
            with open(output_file, 'w', encoding='utf8') as writer:
                for line in reader:
                    words = line.rstrip().split()
                    new_line = ''
                    for word in words:
                        if word in vocab_set:
                            new_line = new_line + ' ' + word
                        else:
                            new_line = new_line + ' ' + unk
                    writer.write(new_line + '\n')

def main():
    util = TheEdgeCorpusUtils()
    util.save_info('../data/theEdge_2018-2021/', 'edge_raw.csv')
    util.annotate_company('edge_raw.csv', 'stock_listed.csv', 'edge_labeled.csv')
    util.setup_data('edge_labeled.csv', 'stock_listed.csv', '../Trading/data/KLSE', 20120103, 20210319,
                    'edge_complete.csv')
    util.prepare_data('edge.csv', 'edge-text.txt', 'edge-company.txt', 'edge-label.txt')

    m_util = MultiSourceClassifierUtils()
    m_util.tokenize_text('data/train.unnorm.txt', 'data/train.norm.txt', 'english')
    m_util.tokenize_text('data/test.unnorm.txt', 'data/test.norm.txt', 'english')
    m_util.tokenize_text('data/valid.unnorm.txt', 'data/valid.norm.txt', 'english')

    # normalize the digit: ./preprocess.sh
    # convert UNK: ./convert_UNK.py
    util.convert_to_unk('data/train.norm.txt', 'edgeMarket.vocab', 'data/train.txt')
    util.convert_to_unk('data/test.norm.txt', 'edgeMarket.vocab', 'data/test.txt')
    util.convert_to_unk('data/valid.norm.txt', 'edgeMarket.vocab', 'data/valid.txt')



if __name__ == '__main__':
    main()
