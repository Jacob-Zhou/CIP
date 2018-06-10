import datetime
import sys
import numpy as np


class dataset(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        f = open(filename, encoding='utf-8')
        while (True):
            line = f.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
                word_num += 1
        self.sentences_num = len(self.sentences)
        self.word_num = word_num

        print('%s:共%d个句子,共%d个词。' % (filename, self.sentences_num, self.word_num))
        f.close()


class liner_model(object):
    def __init__(self):
        self.train_data = dataset('./data/train.conll')
        self.dev_data = dataset('./data/dev.conll')
        self.features = {}
        self.weights = []
        self.v = []
        self.tag_dict = {}

    def create_feature_template(self, sentence, position):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_word)
        template.append('03:' + last_word)
        template.append('04:' + next_word)
        template.append('05:' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_word_first_char)
        template.append('08:' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + sentence[position][i])
            template.append('10:' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + sentence[position][-1] + '*' + sentence[position][i])

        if len(sentence[position]) == 1:
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, len(sentence[position]) - 1):
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + sentence[position][i] + '*' + 'consecutive')

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + sentence[position][0:i + 1])
            template.append('15:' + sentence[position][-(i + 1)::])

        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                template = self.create_feature_template(sentence, j)
                for f in template:
                    if f not in self.features.keys():
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tag_dict.keys():
                        self.tag_dict[tag] = len(self.tag_dict)

        self.weights = [0] * (len(self.features) * len(self.tag_dict))
        self.v = [0] * (len(self.features) * len(self.tag_dict))
        self.update_times = [0] * (len(self.features) * len(self.tag_dict))
        print("the total number of features is %d" % (len(self.features)))

    def dot(self, feature_id, tag, averaged=False):
        score = 0
        offset = self.tag_dict[tag] * len(self.features)
        for f_id in feature_id:
            if averaged:
                score += self.v[f_id + offset]
            else:
                score += self.weights[f_id + offset]
        return score

    def predict(self, sentence, position, averaged=False):
        template = self.create_feature_template(sentence, position)
        f_id = []
        for f in template:
            if f in self.features.keys():
                f_id.append(self.features[f])
        tag_id = np.argmax([self.dot(f_id, tag, averaged) for tag in self.tag_dict])
        return list(self.tag_dict)[tag_id]

    def save(self, path):
        f = open(path, 'w', encoding='utf-8')
        for tag, id in self.tag_dict.items():
            offset = id * len(self.features)
            for feature, feature_id in self.features.items():
                f.write(feature[0:3] + tag + '*' + feature[3:] + '\t' + str(self.weights[offset + feature_id]) + '\n')
        f.close()

    def evaluate(self, data, averaged=False):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            for j in range(len(sentence)):
                predict_tag = self.predict(sentence, j, averaged)
                if predict_tag == tags[j]:
                    correct_num += 1

        return (correct_num, total_num, correct_num / total_num)

    def train(self, averaged=False):
        max_dev_precision = 0.0
        max_iterator = -1
        update_time = 0
        if averaged == True:
            print('using V to predict dev data...')
        for iterator in range(20):
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                for j in range(len(sentence)):
                    predict_tag = self.predict(sentence, j, False)
                    gold_tag = tags[j]

                    if predict_tag != gold_tag:
                        update_time += 1
                        feature = self.create_feature_template(sentence, j)
                        offset_predict_tag = self.tag_dict[predict_tag] * len(self.features)
                        offset_gold_tag = self.tag_dict[gold_tag] * len(self.features)

                        for f in feature:
                            if f in self.features.keys():
                                index = self.features[f] + offset_predict_tag
                                last_w_value = self.weights[index]
                                self.weights[index] -= 1
                                self.update_v(index, update_time, last_w_value)

                                index = self.features[f] + offset_gold_tag
                                last_w_value = self.weights[index]
                                self.weights[index] += 1
                                self.update_v(index, update_time, last_w_value)
            # 本次迭代完成
            current_update_times = update_time  # 本次更新所在的次数
            for i in range(len(self.v)):
                last_w_value = self.weights[i]
                last_update_times = self.update_times[i]  # 上一次更新所在的次数
                if (current_update_times != last_update_times):
                    self.update_times[i] = current_update_times
                    self.v[i] += (current_update_times - last_update_times - 1) * last_w_value + self.weights[i]

            print('iterator: %d' % (iterator))
            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data, averaged)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))
            if dev_precision > (max_dev_precision + 1e-10):
                max_dev_precision = dev_precision
                max_iterator = iterator
                # self.save('./result.txt')
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision))

    def update_v(self, index, update_time, last_w_value):
        last_update_time = self.update_times[index]  # 上一次更新所在的次数
        current_update_time = update_time  # 本次更新所在的次数
        self.update_times[index] = update_time
        self.v[index] += (current_update_time - last_update_time - 1) * last_w_value + self.weights[index]


if __name__ == '__main__':
    if len(sys.argv) == 2:
        averaged = sys.argv[1]
    else:
        averaged = False
    starttime = datetime.datetime.now()
    lm = liner_model()
    lm.create_feature_space()
    if averaged == 'averaged':
        lm.train(averaged=True)
    else:
        lm.train()
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
