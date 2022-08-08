import preprocess
import embeddings_algorithms as embeddings
import postprocess
import time
import datetime
import csv
from tqdm import tqdm
import math
import graph_analysis as graph


def main ():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    main_path = r'C:\Users\CFSM\Desktop\Embeddings'
    data_path = main_path + r'\data'
    training_path = data_path + r'\aclImdb\train'                   
    sk_projector_path = main_path + r'\projector\sk_'
    sk_logs_path = main_path + r'\logs\sk'
    st_projector_path = main_path + r'\projector\st_'
    st_logs_path = main_path + r'\logs\st'
    dp_projector_path = main_path + r'\projector\dp_'
    dp_logs_path = main_path + r'\logs\dp'
    tests_path = main_path + r'\tests\sk'
    
    # PREPROCESS
    if(preprocess.download_aclImdb(url, data_path)):
        preprocess.clean_aclImdb_download(data_path)
        preprocess.filter_data(data_path + r'\aclImdb')
    
    # ALGORITHM TESTS 
    tf_train, tf_text, tf_val = preprocess.get_tf_data(training_path)
    sk_train = preprocess.get_sk_data(training_path)
    trunc_size = 20
    increment = 500
    vocab, _ = embeddings.run_sk(sk_train, None, trunc_size)
    max_vocab_length = 20000#len(vocab)
    vocab_sizes = []
    max_vocab_size = (math.floor(max_vocab_length / increment) + 1) * increment
    for i in range(increment, max_vocab_size, increment):
        vocab_sizes.append(i)
        
    num_trials = 3
    st_max_vocab = 3000
    test_start_time = time.time()
    test_start_date = datetime.datetime.now().strftime("%Y_%m_%d__%I_%M_%S_%p")
    logname = main_path + r'\timings\timing_log_' + test_start_date  + '.csv'
    with open(logname, 'w', encoding='UTF8', newline = '') as f:
        row =['vocab', 'trial', 'sk', 'st', 'dp']
        writer = csv.writer(f)
        writer.writerow(row)
        for vocab_size in vocab_sizes:
            row[0] = str(vocab_size)
            for i in range(num_trials):
                row[1] = str(i + 1)
                print(f'sk-embeddings started: {i + 1} out of {num_trials}')
                print(f'The number of vocabulary items in the sample: {vocab_size}')
                old_time = time.time()
                vocab, v = embeddings.run_sk(sk_train, vocab_size, trunc_size)
                row[2] = time.time() - old_time
                print('sk-embeddings finished \n')
                if(vocab_size <= st_max_vocab):
                    print(f'st-embeddings started: {i + 1} out of {num_trials}')
                    print(f'The number of vocabulary items in the sample: {vocab_size}')
                    old_time = time.time()
                    vocab, v = embeddings.run_st(tf_text, tf_val, st_logs_path,vocab_size, trunc_size)
                    row[3] = time.time() - old_time
                    print('st-embeddings finished \n')
                else:
                    row[3] = ''
                print(f'dp-embeddings started: {i + 1} out of {num_trials}')
                print(f'The number of vocabulary items in the sample: {vocab_size}')
                old_time = time.time()
                vocab, v = embeddings.run_dp(tf_train, tf_text, tf_val, dp_logs_path,vocab_size, trunc_size)
                row[4] = time.time() - old_time
                print('dp-embeddings finished \n')
                
                writer.writerow(row)
                #row[0] = ''
    
    print(time.time() - test_start_time)            
                
    # POSTPROCESS
    #sk_post = postprocess.Postprocess(vocab, v)
    #sk_post.generate_projector_data(projector_path)
    #sk_post.test_all(tests_path)
    graph_name = main_path + r'\timings\sk_graph_' + test_start_date + '.png'
    graph.GraphAnalysis(logname, 'vocab', 'sk', graph_name).gen_graph()
    graph_name = main_path + r'\timings\dp_graph_' + test_start_date + '.png'
    graph.GraphAnalysis(logname, 'vocab', 'dp', graph_name).gen_graph()
    
if __name__ == '__main__':
    main()
