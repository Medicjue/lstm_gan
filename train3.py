import os
import sys

import tensorflow as tf
import numpy as np

import utils
from lstm_gan import LSTMGAN
import csv

import sys
import codecs
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

outputSentences = []

DATASET_FILE = 'out.pickle'
VOCABULARY_FILE = 'vocabulary.pickle'

SENTENCE_SIZE = 30
VOCABULARY_SIZE = 10000

SENTENCE_START_TOKEN = "START"
SENTENCE_END_TOKEN = "END"

def uniArray(array_unicode):
    items = [x.encode('utf-8') for x in array_unicode]
    array_unicode = np.array([items]) # remove the brackets for line breaks
    return array_unicode

def main():
    args = utils.get_args()
    dataset = utils.load_dataset(os.path.join(args.data_path, DATASET_FILE))
    index2word, word2index = utils.load_dicts(os.path.join(args.data_path, VOCABULARY_FILE))
    
    print("Use dataset with {} sentences".format(dataset.shape[0]))
    
    batch_size = args.batch_size
    noise_size = args.noise_size
    with tf.Graph().as_default(), tf.Session() as session:   
        lstm_gan = LSTMGAN(
            SENTENCE_SIZE,
            len(index2word), ## Using length of vocabularies
            word2index[SENTENCE_START_TOKEN],
            hidden_size_gen = args.hid_gen,
            hidden_size_disc = args.hid_disc,
            input_noise_size = noise_size,
            batch_size = batch_size,
            dropout = args.dropout,
            lr = args.lr,
            grad_cap = args.grad_clip
        )
        
        session.run(tf.initialize_all_variables())

        if args.save_model or args.load_model:
            saver = tf.train.Saver()

        if args.load_model:
            try:
                saver.restore(session, utils.SAVER_FILE)
            except ValueError:
                print("Cant find model file")
                sys.exit(1)
        while True:
            offset = 0.
            for dataset_part in utils.iterate_over_dataset(dataset, batch_size*args.disc_count):
                print("Start train discriminator wih offset {}...".format(offset))
                for ind, batch in enumerate(utils.iterate_over_dataset(dataset_part, batch_size)):
                    noise = np.random.random(size=(batch_size, noise_size))
                    cost = lstm_gan.train_disc_on_batch(session, noise, batch)
                    print("Processed {} sentences with train cost = {}".format((ind+1)*batch_size, cost))

                print("Start train generator...")
                for ind in range(args.gen_count):
                    noise = np.random.random(size=(batch_size, noise_size))
                    cost = lstm_gan.train_gen_on_batch(session, noise)
                    if args.gen_sent:
                        sent = lstm_gan.generate_sent(session, np.random.random(size=(noise_size, )))
                        print(sent)
                        try:
                            sentence = ' '.join(index2word[i] for i in sent)
                            outputSentences.append(' '.join(index2word[i] for i in sent))
                            print(' '.join(index2word[i] for i in sent))
                        except:
                            print('Generating sentence error')
                    print("Processed {} noise inputs with train cost {}".format((ind+1)*batch_size, cost))
                
                np.savetxt('genSentences.txt', uniArray(outputSentences), fmt = '%s')
                if args.save_model:
                    saver.save(session, utils.SAVER_FILE)
                    print("Model saved")
        
if __name__ == "__main__":
    main()
