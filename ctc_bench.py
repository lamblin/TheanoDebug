# coding:utf-8
# Test bench for different CTC implementations
# Author    :  David Leon (Dawei Leng)
# Created   :   1,  7, 2016
# Revised   :   1, 27, 2016
#------------------------------------------------------------------------------------------------
from __future__ import print_function
__author__ = 'dawei.leng'
from ctc_theano import CTC_precise, CTC_for_train
import numpy as np, theano
import theano.tensor as tensor
floatX = theano.config.floatX


if __name__ == '__main__':
    np.random.seed(33)
    C = 10
    L = 50
    T = 200
    blank = C
    B = 2
    x1, x2, x3, x4, x5 = tensor.imatrix(name='queryseq'), tensor.tensor3(dtype=floatX, name='scorematrix'), \
                         tensor.fmatrix(name='queryseq_mask'), tensor.fmatrix(name='scorematrix_mask'), \
                         tensor.iscalar(name='blank_symbol')


    result = CTC_precise.cost(x1, x2, x3, x4, x5)
    f1 = theano.function([x1, x2, x3, x4, x5], result)


    x6 = tensor.zeros([L,B], dtype='int32')
    result2 = CTC_for_train.cost(x6, x2, x3, x4, x5)
    f3 = theano.function([x6, x2, x3, x4, x5], result2)

    theano.printing.debugprint(result2, print_type=True, file=open('f3_out', 'w'))
    theano.printing.debugprint(theano.function([x6, x2, x3, x4, x5], result2, mode=theano.Mode(optimizer=None)),
                               print_type=True, file=open('f3_unopt', 'w'))
    theano.printing.debugprint(f3, print_type=True, file=open('f3_opt', 'w'))

    result2 = CTC_for_train.cost(x1, x2, x3, x4, x5)
    f4 = theano.function([x1, x2, x3, x4, x5], result2)

    theano.printing.debugprint(result2, print_type=True, file=open('f4_out', 'w'))
    theano.printing.debugprint(theano.function([x1, x2, x3, x4, x5], result2, mode=theano.Mode(optimizer=None)),
                               print_type=True, file=open('f4_unopt', 'w'))
    theano.printing.debugprint(f4, print_type=True, file=open('f4', 'w'))

    scorematrix = np.random.randn(C+1, T)
    scorematrix -= np.max(scorematrix, axis=0)
    scorematrix = np.exp(scorematrix)
    scorematrix /= np.sum(scorematrix, axis=0)

    for _ in range(10):

        if 1:
            seq = np.floor(np.random.rand(L, B) * C).astype(np.int32)


            y=seq.reshape([L, B])
            yhat = scorematrix.T.reshape([T, C+1, 1])
            yhat = np.concatenate((yhat, yhat), axis=2)

            NLL_theano_batch = f1(y, yhat, np.ones_like(y, dtype=np.float32), np.ones([T,B], dtype=np.float32), blank)

            NLL_theano_batch_log = f3(y, yhat, np.ones_like(y, dtype=np.float32), np.ones([T,B], dtype=np.float32), blank)

            NLL_theano_batch_log2 = f4(y, yhat, np.ones_like(y, dtype=np.float32), np.ones([T,B], dtype=np.float32), blank)

            print("NLL_theano_batch = %f, NLL_theano_batch_log2 = %f, NLL_theano_batch_log = %f" %
                  (NLL_theano_batch, NLL_theano_batch_log2, NLL_theano_batch_log))
            print()
