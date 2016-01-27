# TheanoDebug
Exclusively for theano debug, issue https://github.com/Theano/Theano/issues/3925

# Problem reproduction
There're two cost functions defined in ctc_theano.py, CTC_precise.cost() and CTC_for_train.cost().
The problem is related to how the input of CTC_for_train.cost() is declared. When the first input of
CTC_for_train.cost() is declared as

    x1 = tensor.imatrix()     # [ctc_bench, Line20]  
      
the cost function will give correct result (same as the result by CTC_precise.cost()). However, if
declared as

    x6 = tensor.zeros([L,B], dtype='int32')    #[ctc_bench, Line29]  
      
the cost function will give incorrect result. Run the ctc_bench.py and you can see the results like:

    NLL_theano_batch = 324.517242, NLL_theano_batch_log2 = 324.517242, NLL_theano_batch_log = 359.621189
    
in which 'NLL_theano_batch_log2' is the result when input declared as x1 for CTC_for_train.cost(), and
'NLL_theano_batch_log' is the result when input delcared as x6.

# Comment
The above problem doesn't affect CTC_precise.cost() function.  
theano version = revision e891fb3, 2016-1-26 12:52:44  
python version = 3.5.0  
platform = windows X64