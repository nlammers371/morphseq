#!/usr/bin/env python

import os, sys, argparse, random
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeCV
import scipy
from scipy.io import mmread

sys.path.append('bin/')
from train_scRNA_vae import scRNA_VAE
from evaluation_functions import *

def load_rna_file_sparse(url1):
    """
    load scRNA mtx file to sparse matrix, binarize the data; load batch info
    """
    #logger.info('url1={}'.format(url1)); assert os.path.isfile(url1);
    data_rna = mmread(url1).tocsr()
    data_rna = data_rna.astype(int)
    data_rna_batch = mmread(url1.split('.mtx')[0]+'_barcodes_dataset.mtx').tocsr()
    return data_rna, data_rna_batch

'''
def pred_atac_norm(autoencoder, data_rna_test, batch_rna_test, input_dim_atac, output_prefix='', atac_index='',
                   batch_size=16):
    """
    predict scATAC normalized expression from scRNA input
    Parameters
    ----------
    data_rna_test: scRNA expression, ncell x input_dim_rna
    batch_rna_test: scRNA batch factor, ncell x batch_dim_x
    input_dim_atac: scATAC batch factor dimension, int
    output_prefix: output prefix
    atac_index: index of scATAC peak to output ('' for all peaks)

    Output
    ----------
    test_translator_reconstr_atac_norm_output: ncell x npeaks
    """
    test_translator_reconstr_atac_norm = {}
    for batch_id in range(0, data_rna_test.shape[0] // batch_size + 1):
        index_range = list(range(batch_size * batch_id, min(batch_size * (batch_id + 1), data_rna_test.shape[0])))
        test_translator_reconstr_atac_norm[batch_id] = autoencoder.predict_atacnorm_translation(
            data_rna_test[index_range,].todense(), batch_rna_test[index_range,].todense(), input_dim_atac);

    test_translator_reconstr_atac_norm_output = np.concatenate(
        [v for k, v in sorted(test_translator_reconstr_atac_norm.items())], axis=0)
    if atac_index != '':
        test_translator_reconstr_atac_norm_output = test_translator_reconstr_atac_norm_output[:, atac_index]
    if output_prefix == '':
        return (test_translator_reconstr_atac_norm_output)
    else:
        np.savetxt(output_prefix + '_atacnorm_pred.txt', test_translator_reconstr_atac_norm_output, delimiter='\t',
                   fmt='%1.5f')
'''

def pred_latent_scRNA(autoencoder, data_rna_test, batch_rna_test, output_prefix='', batch_size=16):
    """
    predict scATAC normalized expression from scRNA input
    Parameters
    ----------
    data_rna_test: scRNA expression, ncell x input_dim_rna
    batch_rna_test: scRNA batch factor, ncell x batch_dim_x
    output_prefix: output prefix

    Output
    ----------
    test_translator_reconstr_atac_norm_output: ncell x npeaks
    """
    test_translator_reconstr_atac_norm = {}
    for batch_id in range(0, data_rna_test.shape[0] // batch_size + 1):
        index_range = list(range(batch_size * batch_id, min(batch_size * (batch_id + 1), data_rna_test.shape[0])))
        test_translator_reconstr_atac_norm[batch_id] = autoencoder.predict_atacnorm_translation(
            data_rna_test[index_range,].todense(), batch_rna_test[index_range,].todense(), input_dim_atac);

    test_translator_reconstr_atac_norm_output = np.concatenate(
        [v for k, v in sorted(test_translator_reconstr_atac_norm.items())], axis=0)
    if output_prefix == '':
        return (test_translator_reconstr_atac_norm_output)
    else:
        np.savetxt(output_prefix + '_atacnorm_pred.txt', test_translator_reconstr_atac_norm_output, delimiter='\t',
                   fmt='%1.5f')

'''
def eval_rna_correlation(autoencoder, data_rna_val, data_atac_val, batch_atac_val, data_rna_test, data_atac_test, batch_atac_test, batch_dim_x, output_prefix, batch_size=16):
    """
    evaluate predicted scRNA normalized expression for both validation and test set, and return gene-wise correlation
    Parameters
    ----------
    data_atac_test: scATAC expression, ncell x input_dim_atac
    batch_atac_test: scATAC batch factor, ncell x batch_dim_y
    output_prefix: output prefix

    """
    sim_metric_rna = []
    val_translation_rna_norm = pred_rna_norm(autoencoder, data_rna_val.shape[1], data_atac_val, batch_atac_val)
    cor_gene, cor_gene_flatten, npos_gene = plot_cor_pergene(data_rna_val.todense(), val_translation_rna_norm, logscale=True, normlib='norm')
    sim_metric_rna.extend([np.nanmean(cor_gene), cor_gene_flatten])
    test_translation_rna_norm = pred_rna_norm(autoencoder, data_rna_val.shape[1], data_atac_test, batch_atac_test)
    cor_gene, cor_gene_flatten, npos_gene = plot_cor_pergene(data_rna_test.todense(), test_translation_rna_norm, logscale=True, normlib='norm')
    sim_metric_rna.extend([np.nanmean(cor_gene), cor_gene_flatten])
    np.savetxt(output_prefix+'_test_rna_cor.txt', cor_gene, delimiter='\n', fmt='%1.5f')
    np.savetxt(output_prefix+'_stats_rna_cor.txt', sim_metric_rna, delimiter='\n', fmt='%1.5f')

'''

def train_scRNA_vae(outdir, sim_url, train_test_split, path_x, path_x_single, dispersion,
                          embed_dim_x, nlayer, dropout_rate, learning_rate_x,
                          trans_ver, hidden_frac, kl_weight, patience,
                          nepoch_warmup_x, nepoch_klstart_x, batch_size, train,
                          evaluate, predict):
    """
    train/load the Polarbear model
    Parameters
    ----------
    data_rna_train: scRNA training data, ncell x input_dim_rna
    data_rna_val: scRNA validation data, ncell x input_dim_rna
    data_rna_test: scRNA data, ncell x input_dim_rna
    batch_rna_val: scRNA batch matrix, ncell x nbatch
    outdir: output directory
    train_test_split: "random" or "babel"
    path_x: scRNA co-assay (SNARE-seq) file path
    path_x_single: scRNA single-assay file path
    train: "train" or "predict", train the model or just load existing model

    """
    os.system('mkdir -p ' + outdir)

    data_rna, data_rna_batch = load_rna_file_sparse(path_x)

    ## ======================================
    ## define train, validation and test
    data_rna_barcode = pd.read_csv(path_x.split('.mtx')[0] + '_barcodes.tsv', delimiter='\t')
    barcode_list = data_rna_barcode['index'].to_list()

    ## randomly assign 1/5 as validation and 1/5 as test set
    cv_size = data_rna.shape[0] // 5
    rand_index = list(range(data_rna.shape[0]))
    random.seed(101)
    random.shuffle(rand_index)
    all_ord_index = range(data_rna.shape[0])
    test_index = rand_index[0: cv_size]
    all_ord_index = list(set(all_ord_index) - set(range(0, cv_size)))
    random.seed(101)
    val_ord_index = random.sample(all_ord_index, cv_size)
    all_ord_index = list(set(all_ord_index) - set(val_ord_index))
    val_index = [rand_index[i] for i in val_ord_index]
    train_index = [rand_index[i] for i in all_ord_index]
    data_rna_train = data_rna[train_index,]
    batch_rna_train = data_rna_batch[train_index,]
    data_rna_train_co = data_rna[train_index,]
    batch_rna_train_co = data_rna_batch[train_index,]
    data_rna_test = data_rna[test_index,]
    batch_rna_test = data_rna_batch[test_index,]
    data_rna_val = data_rna[val_index,]
    batch_rna_val = data_rna_batch[val_index,]

    if train == 'train':
        ## save the corresponding barcodes
        train_barcode = np.array(barcode_list)[train_index]
        valid_barcode = np.array(barcode_list)[val_index]
        test_barcode = np.array(barcode_list)[test_index]
        np.savetxt(sim_url + '_train_barcodes.txt', train_barcode, delimiter='\n', fmt='%s')
        np.savetxt(sim_url + '_valid_barcodes.txt', valid_barcode, delimiter='\n', fmt='%s')
        np.savetxt(sim_url + '_test_barcodes.txt', test_barcode, delimiter='\n', fmt='%s')

        ## load single assay data
        if path_x_single != 'nornasingle':
            data_rna_single, data_rna_single_batch = load_rna_file_sparse(path_x_single)
            data_rna_train = scipy.sparse.vstack((data_rna_train, data_rna_single))
            batch_rna_train = scipy.sparse.vstack((batch_rna_train, data_rna_single_batch))

        ## shuffle training set
        rand_index_rna = list(range(data_rna_train.shape[0]))
        random.seed(101)
        random.shuffle(rand_index_rna)
        data_rna_train = data_rna_train[rand_index_rna,]
        batch_rna_train = batch_rna_train[rand_index_rna,]

        ## train the model
        tf.reset_default_graph()

        autoencoder_rna = scRNA_VAE(input_dim_x=data_rna_val.shape[1], batch_dim_x=batch_rna_val.shape[1],
                                  embed_dim_x=embed_dim_x, dispersion=dispersion, nlayer=nlayer, dropout_rate=dropout_rate,
                                  learning_rate_x=learning_rate_x,hidden_frac=hidden_frac, kl_weight=kl_weight)

        autoencoder_rna.train(data_rna_train, batch_rna_train, data_rna_val, batch_rna_val, data_rna_train_co,
                              batch_rna_train_co, nepoch_warmup_x, patience, nepoch_klstart_x, output_model=sim_url,
                              batch_size=batch_size, nlayer=nlayer, save_model=True)

    elif train == 'predict':
        tf.reset_default_graph()
        autoencoder_rna = scRNA_VAE(input_dim_x=data_rna_val.shape[1], batch_dim_x=batch_rna_val.shape[1],
                                    embed_dim_x=embed_dim_x, dispersion=dispersion, nlayer=nlayer,
                                    dropout_rate=dropout_rate,
                                    learning_rate_x=learning_rate_x, hidden_frac=hidden_frac, kl_weight=kl_weight)

        autoencoder_rna.load(sim_url);
    '''
    if evaluate == 'evaluate':
        ## evaluate on size-normalized scRNA true profile
        output_prefix = sim_url
        eval_rna_correlation(autoencoder, data_rna_val, data_atac_val, batch_atac_val, data_rna_test, data_atac_test,
                             batch_atac_test, batch_rna_train.shape[1], output_prefix)

        ## evaluate on true scATAC profile
        # we report performance on a subset of peaks that are differentially expressed across cell types based on the SNARE-seq study (Chen et al. 2019)
        chr_annot_selected = pd.read_csv(args.path_y.split('snareseq')[0] + 'peaks_noXY_diffexp.txt', sep='\t')
        atac_index = chr_annot_selected['index'].tolist()
        atac_index[:] = [int(number) - 1 for number in atac_index]
        eval_atac_AUROC_AUPR(autoencoder, data_rna[train_index,], data_atac[train_index,], data_rna_val, data_atac_val,
                             batch_rna_val, data_rna_test, data_atac_test, batch_rna_test, batch_atac_train.shape[1],
                             output_prefix, atac_index)

        ## evaluate alignment
        eval_alignment(autoencoder, data_rna_val, data_atac_val, batch_rna_val, batch_atac_val, data_rna_test,
                       data_atac_test, batch_rna_test, batch_atac_test, output_prefix)
   
    if predict == 'predict':
        ## output normalized scRNA prediction
        output_prefix = sim_url + '_test'
        pred_rna_norm(autoencoder, data_rna_train.shape[1], data_atac_test, batch_atac_test,
                      output_prefix=output_prefix)
        if train_test_split == 'babel':
            ## output prediction on the training set, to compare with the unseen cell type prediction
            output_prefix = sim_url + '_train'
            pred_rna_norm(autoencoder, data_rna_train.shape[1], data_atac[train_index,], data_atac_batch[train_index,],
                          output_prefix=output_prefix)

        ## output normalized scATAC prediction on the test set
        if False:
            chr_annot_selected = pd.read_csv(args.path_y.split('snareseq')[0] + 'peaks_noXY_diffexp.txt', sep='\t')
            atac_index = chr_annot_selected['index'].tolist()
            atac_index[:] = [int(number) - 1 for number in atac_index]
            pred_atac_norm(autoencoder, data_rna_test, batch_rna_test, data_atac_train.shape[1], output_prefix,
                           atac_index)

        ## output alignment on the test set
        pred_embedding(autoencoder, data_rna_test, batch_rna_test, data_atac_test, batch_atac_test, output_prefix)
     '''

def main(args):
    learning_rate_x = args.learning_rate_x;
    learning_rate_y = args.learning_rate_y;
    learning_rate_xy = args.learning_rate_xy;
    learning_rate_yx = args.learning_rate_yx;
    embed_dim_x = args.embed_dim_x;
    embed_dim_y = args.embed_dim_y;
    dropout_rate = float(args.dropout_rate);
    nlayer = args.nlayer;
    batch_size = args.batch_size
    trans_ver = args.trans_ver
    patience = args.patience
    nepoch_warmup_x = args.nepoch_warmup_x
    nepoch_warmup_y = args.nepoch_warmup_y
    nepoch_klstart_x = args.nepoch_klstart_x
    nepoch_klstart_y = args.nepoch_klstart_y
    dispersion = args.dispersion
    hidden_frac = args.hidden_frac
    kl_weight = args.kl_weight
    train_test_split = args.train_test_split

    sim_url = args.outdir + 'polarbear_' + train_test_split + '_' + dispersion + '_' + str(nlayer) + 'l_lr' + str(
        learning_rate_y) + '_' + str(learning_rate_x) + '_' + str(learning_rate_xy) + '_' + str(
        learning_rate_yx) + '_dropout' + str(dropout_rate) + '_ndim' + str(embed_dim_x) + '_' + str(
        embed_dim_y) + '_batch' + str(batch_size) + '_' + trans_ver + '_improvement' + str(
        patience) + '_nwarmup_' + str(nepoch_warmup_x) + '_' + str(nepoch_warmup_y) + '_klstart' + str(
        nepoch_klstart_x) + '_' + str(nepoch_klstart_y) + '_klweight' + str(kl_weight) + '_hiddenfrac' + str(
        hidden_frac)
    print(sim_url)
    train_polarbear_model(args.outdir, sim_url, train_test_split, args.path_x, args.path_y, args.path_x_single,
                          args.path_y_single, dispersion, embed_dim_x, embed_dim_y, nlayer, dropout_rate,
                          learning_rate_x, learning_rate_y, learning_rate_xy, learning_rate_yx, trans_ver, hidden_frac,
                          kl_weight, patience, nepoch_warmup_x, nepoch_warmup_y, nepoch_klstart_x, nepoch_klstart_y,
                          batch_size, args.train, args.evaluate, args.predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--train_test_split', type=str, help='train/val/test split version, "random" or "babel"',
                        default='random');
    parser.add_argument('--train', type=str,
                        help='"train": train the model from the beginning; "predict": load existing model for downstream prediction',
                        default='predict');
    parser.add_argument('--evaluate', type=str,
                        help='"evaluate": evaluate translation and alignment performance on the validation set',
                        default='');
    parser.add_argument('--predict', type=str, help='"predict": predict translation and alignment on test set',
                        default='');

    parser.add_argument('--path_x_single', type=str, help='path of scRNA single assay data file',
                        default='nornasingle');
    parser.add_argument('--path_y_single', type=str, help='path of scATAC single assay data file',
                        default='noatacsingle');
    parser.add_argument('--path_x', type=str, help='path of scRNA snare-seq co-assay data file');
    parser.add_argument('--path_y', type=str, help='path of scATAC snare-seq co-assay data file');
    parser.add_argument('--nlayer', type=int, help='number of hidden layers in encoder and decoders in neural network',
                        default=2);
    parser.add_argument('--outdir', type=str, help='outdir', default='./');
    parser.add_argument('--batch_size', type=int, help='batch size', default=16);
    parser.add_argument('--learning_rate_x', type=float, help='scRNA VAE learning rate', default=0.001);
    parser.add_argument('--learning_rate_y', type=float, help='scATAC VAE learning rate', default=0.0001);
    parser.add_argument('--learning_rate_xy', type=float,
                        help='scRNA embedding to scATAC embedding translation learning rate', default=0.001);
    parser.add_argument('--learning_rate_yx', type=float,
                        help='scATAC embedding to scRNA embedding translation learning rate', default=0.001);
    parser.add_argument('--dropout_rate', type=float, help='dropout rate in VAE', default=0.1);
    parser.add_argument('--embed_dim_x', type=int, help='embed_dim_x', default=25);
    parser.add_argument('--embed_dim_y', type=int, help='embed_dim_y', default=25);
    parser.add_argument('--trans_ver', type=str, help='translation layer in between embeddings, linear or 1l or 2l',
                        default='linear');
    parser.add_argument('--patience', type=int, help='patience for early stopping', default=45);
    parser.add_argument('--nepoch_warmup_x', type=int,
                        help='number of epochs to take to warm up RNA VAE kl term to maximum', default=400);
    parser.add_argument('--nepoch_warmup_y', type=int,
                        help='number of epochs to take to warm up ATAC VAE kl term to maximum', default=80);
    parser.add_argument('--nepoch_klstart_x', type=int,
                        help='number of epochs to wait to start to warm up RNA VAE kl term', default=0);
    parser.add_argument('--nepoch_klstart_y', type=int,
                        help='number of epochs to wait to start to warm up ATAC VAE kl term', default=0);
    parser.add_argument('--dispersion', type=str,
                        help='estimate dispersion per gene&batch: genebatch or per gene&cell: genecell',
                        default='genebatch');
    parser.add_argument('--hidden_frac', type=int, help='shrink intermediate dimension by dividing this term',
                        default=2);
    parser.add_argument('--kl_weight', type=float, help='weight of kl loss in beta-VAE', default=1);

    args = parser.parse_args();
    main(args);