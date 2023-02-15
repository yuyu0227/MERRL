import argparse




def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="NEAC")
    parser.add_argument("--log_dir", type=str, default="log")

    parser.add_argument("--dump_path", type=str, default="./", help="Experiment saved root path")
    

    parser.add_argument("--task", type=str, default="sentiment", help="Downstream NLP task")
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--ckpt", type=str, default="", help="reload path for pre-trained model")
    parser.add_argument("--seed", type=int, default=555, help="random seed (five seeds: 555, 666, 777, 888, 999)")
    parser.add_argument("--tgt_dm", type=str, default="politics", help="target domain")
    parser.add_argument("--rand_sel_ratio", type=float, default=0, help="random selection ratio e.g. 0.5, 0.6")

    # nlp train parameters
    parser.add_argument("--emb_file", type=str, default="", help="embeddings file eg.(glove.6B.300d.txt)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=2, help="No improvement after several epoch, we stop training")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    

    
    #NER
    parser.add_argument("--num_tag", type=int, default=0, help="Number of entity in the dataset")
    parser.add_argument("--bert_hidden_dim", type=int, default=768, help="Hidden layer dimension for bert")

    #POS Tagging
    parser.add_argument("--use_bert", type=bool, default=False, help="whether use bert tagger")
    parser.add_argument("--use_brown", type=bool, default=True, help="whether use brown corpus")
    parser.add_argument("--lstm_emb_dim", type=int, default=100, help="embedding dimension")
    parser.add_argument("--n_layers", type=int, default=1, help="number of layers for LSTM")
    parser.add_argument("--lstm_hidden_dim", type=int, default=100, help="embedding dimension")
    
    #Sentiment Classification
    parser.add_argument("--cnn_emb_dim", type=int, default=300, help="embedding dimension for CNN")
    parser.add_argument("--n_filters", type=int, default=100, help="number of filters for CNN")
    parser.add_argument("--filter_sizes", type=list, default=[3,4,5], help="filter sizes for CNN")
    
    #conversation
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    #parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--adam_betas', default='(0.9,0.999)')
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--num_warmup_steps', default=500, type=int)
    parser.add_argument('--max_length', default=128, type=int)

    parser.add_argument('--accumulate_grad_batches', default=4, type=int)
    parser.add_argument('--gradient_clip_val', default=0.1, type=float)
    parser.add_argument('--max_epochs', default=16, type=int)

    #RL training 
    parser.add_argument("--alg", type=str, default="baseline", help="e.g. baseline, random_selection, reinforce, a2c, ppo..")
    parser.add_argument("--bag_size", type=int, default=200, help="The size of data we choose subset from")
    parser.add_argument("--inner_epoch", type=int, default=1, help="The epoch to train nlp model after selecting a subset of data from a bag of data")
    parser.add_argument("--pretrain_epoch", type=int, default=2, help="The epoch to train nlp model before training on rl model")
    parser.add_argument("--rl_timesteps", type=int, default=200, help="total training timesteps for rl model")
    parser.add_argument("--reward_mode", type=str, default="shannon_entropy", help="the way to compute reward for rl model")
    
    params = parser.parse_args()
    return params
