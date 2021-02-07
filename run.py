# -*- coding:utf-8 -*-
import os
import numpy as np
import argparse
from NMTM import NMTM
from utils import Data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--e1', type=int, default=100)
parser.add_argument('--e2', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--topic_num', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--test_index', type=int, default=0)
args = parser.parse_args()

print("============args==============")
print(args)

# SEED = 10
# np.random.seed(SEED)
# tf.set_random_seed(SEED)


def print_top_words(beta, id2word, lang, n_top_words=15):
    top_words = []
    for i in range(len(beta)):
        top_words.append(" ".join([id2word[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))

    with open(os.path.join(args.output_dir, 'top_words_T{}_K{}_{}_{}th'.format(n_top_words, args.topic_num, lang, args.test_index)), 'w') as file:
        for line in top_words:
            file.write(line + '\n')
            print(line)


def export_beta(model, data):
    beta_en, beta_cn = model.sess.run([model.beta_en, model.beta_cn])
    print_top_words(beta_en, data.id2word_en, lang='en')
    print_top_words(beta_cn, data.id2word_cn, lang='cn')


def model_test(model, bow_matrix, lang):
    data_size = len(bow_matrix)
    test_loss = np.zeros((data_size,))
    theta = np.zeros((data_size, args.topic_num))

    var_list = [getattr(model, 'loss_{}'.format(lang)), getattr(model, 'z_{}'.format(lang))]
    for i in range(int(data_size / args.batch_size)):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        batch_bow = bow_matrix[start:end]
        feed_dict = {getattr(model, 'x_{}'.format(lang)): batch_bow}
        batch_loss, batch_z = model.sess.run(var_list, feed_dict=feed_dict)
        test_loss[start:end] = batch_loss  #/ np.sum(batch_bow, axis=1)
        theta[start:end] = batch_z

    batch_bow = bow_matrix[-args.batch_size:]
    feed_dict = {getattr(model, 'x_{}'.format(lang)): batch_bow}
    batch_loss, batch_z = model.sess.run(var_list, feed_dict=feed_dict)
    test_loss[-args.batch_size:] = batch_loss  #/ np.sum(batch_bow, axis=1)
    theta[-args.batch_size:] = batch_z

    return test_loss, theta


def export_theta(model, data):
    train_loss_en, train_theta_en = model_test(model, data.train_bow_matrix_en, lang='en')
    train_loss_cn, train_theta_cn = model_test(model, data.train_bow_matrix_cn, lang='cn')

    test_loss_en, test_theta_en = model_test(model, data.test_bow_matrix_en, lang='en')
    test_loss_cn, test_theta_cn = model_test(model, data.test_bow_matrix_cn, lang='cn')

    return train_theta_en, train_theta_cn, test_theta_en, test_theta_cn


def train(model, data):
    total_batch = max(int(data.train_size_cn / args.batch_size), int(data.train_size_en / args.batch_size))
    lr = args.learning_rate
    for epoch in range(args.epoch):
        train_loss = []
        for i in range(total_batch):
            batch_data_en, batch_data_cn = data.load_train_next_batch()
            feed_dict = {model.x_en: batch_data_en, model.x_cn: batch_data_cn, model.keep_prob: args.keep_prob, model.learning_rate: lr}
            _, loss = model.sess.run([model.optimizer, model.loss, ], feed_dict=feed_dict)

            train_loss.append(np.mean(loss))

        print("Epoch: {:3d} train_loss: {:.5f}".format(epoch, np.mean(train_loss)))


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)

    data = Data.TextData(args.data_dir, args.batch_size)

    config = dict()
    config.update(vars(args))

    config['vocab_size_en'] = data.vocab_size_en
    config['vocab_size_cn'] = data.vocab_size_cn

    model = NMTM(config=config, Map_en2cn=data.Map_en2cn, Map_cn2en=data.Map_cn2en)

    train(model, data)
    export_beta(model, data)

    train_theta_en, train_theta_cn, test_theta_en, test_theta_cn = export_theta(model, data)
    np.save(os.path.join(args.output_dir, 'train_theta_K{}_en_{}th.npy'.format(args.topic_num, args.test_index)), train_theta_en)
    np.save(os.path.join(args.output_dir, 'train_theta_K{}_cn_{}th.npy'.format(args.topic_num, args.test_index)), train_theta_cn)
    np.save(os.path.join(args.output_dir, 'test_theta_K{}_en_{}th.npy'.format(args.topic_num, args.test_index)), test_theta_en)
    np.save(os.path.join(args.output_dir, 'test_theta_K{}_cn_{}th.npy'.format(args.topic_num, args.test_index)), test_theta_cn)

    W_en, W_cn = model.sess.run([model.W_en, model.W_cn])
    np.save(os.path.join(args.output_dir, 'W_K{}_en_{}th.npy'.format(args.topic_num, args.test_index)), W_en)
    np.save(os.path.join(args.output_dir, 'W_K{}_cn_{}th.npy'.format(args.topic_num, args.test_index)), W_cn)
