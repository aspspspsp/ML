import csv

class DataManager():
    data_path = ''
    look_up_table_file_name = ''

    # train, test, dev file name
    train_data_file_name = ''
    test_data_file_name = ''
    dev_data_file_name = ''

    # store data for train, test, dev
    train_data = {}
    train_data_vet = {}
    test_data = {}
    dev_data = {}

    # store word vector
    look_up_table = {}

    def __init__(self, data_path, look_up_table_file_name, train_data_file_name, test_data_file_name, dev_data_file_name):
        self.data_path = data_path
        self.look_up_table_file_name = look_up_table_file_name
        self.train_data_file_name = train_data_file_name
        self.test_data_file_name = test_data_file_name
        self.dev_data_file_name = dev_data_file_name

        print('Loading look up table:' + self.look_up_table_file_name)
        self.load_look_up_table()

        print('Loading train data:' + train_data_file_name)
        self.load_train_data()
        #print('Loading test data:' + test_data_file_name)
        #self.load_test_data()
        #print('Loading dev data:' + dev_data_file_name)
        #self.load_dev_data()

    def load_look_up_table(self):
        f = open(self.data_path + self.look_up_table_file_name, 'r')
        for line in f:
            splits = line.split()
            self.look_up_table[splits[0]] = [splits[x] for x in range(1, len(splits))]

        f.close()
        return

    def load_train_data(self):
        self.train_data, self.train_data_vet = self.load_data(self.train_data_file_name)
        return

    def load_test_data(self):
        self.test_data = self.load_data(self.test_data_file_name)
        return

    def load_dev_data(self):
        self.dev_data = self.load_data(self.dev_data_file_name)
        return

    def load_data(self, file_name):
        data = {}
        data_vect = {}

        with open(self.data_path + file_name, 'r', newline='' , errors='ignore') as csvfile:

            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                split = row[1].split()

                data[row[0]] = [split[x] for x in range(1, len(split))]
                data_vect[row[0]] = [self.look_up_word2vec(split[x]) for x in range(1, len(split))]
        return data, data_vect

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return

    def look_up_word2vec(self, word):
        if(word in self.look_up_table):
            return self.look_up_table[word]
        else:
            return None

    def test(self):
        return

'''
def main():
    data_path = 'data/'
    look_up_table_file_name = 'glove.840B.300d.txt'
    train_data_file_name = '07.csv'
    test_data_file_name = ''
    dev_data_file_name = ''

    data_manager = DataManager(data_path, look_up_table_file_name, train_data_file_name, test_data_file_name, dev_data_file_name)

main()
'''