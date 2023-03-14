import argparse
import numpy as np
import time
import torch
import utils
import os
from model import RENet
from global_model import RENet_global
import pickle


def train(args):
    # load data
    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    if args.dataset == 'icews_know':
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
    else:
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'valid.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt','test.txt')

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    use_libxsmm = torch.cuda.is_available()

    print("use_cuda: " + str(use_cuda))
    print("use_libxsmm: " + str(use_libxsmm))
    
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/'+ args.dataset, exist_ok=True)

    model_state_global_file = 'models/' + args.dataset + '/max' + str(args.maxpool) + 'rgcn_global.pth'

    epoch = args.start_epoch

    print("start validation...")

    best_mrr = 0
    while True:
        if epoch == args.max_epochs:
            break
        epoch += 1
        
        model = RENet(num_nodes,
                        args.n_hidden,
                        num_rels,
                        dropout=args.dropout,
                        model=args.model,
                        seq_len=args.seq_len,
                        num_k=args.num_k, use_cuda=use_cuda)
        global_model = RENet_global(num_nodes,
                            args.n_hidden,
                            num_rels,
                            dropout=args.dropout,
                            model=args.model,
                            seq_len=args.seq_len,
                            num_k=args.num_k, maxpool=args.maxpool, use_cuda=use_cuda, use_libxsmm=use_libxsmm)
        
        epoch_model_state_file = 'models/' + args.dataset + "/" + str(epoch) + "/rgcn.pth"
        epoch_model_state_global_file2 = 'models/' + args.dataset + "/" + str(epoch) + "/max" + str(args.maxpool) + "rgcn_global2.pth"

        print("loading " + epoch_model_state_file)
        print("loading " + epoch_model_state_global_file2)
        model_dict = torch.load(epoch_model_state_file)
        global_model_dict = torch.load(epoch_model_state_global_file2)

        model.load_state_dict(model_dict['state_dict'])
        global_model.load_state_dict(global_model_dict['state_dict'])
    
        checkpoint_global = torch.load(model_state_global_file, map_location=lambda storage, loc: storage)
        global_model.load_state_dict(checkpoint_global['state_dict'])
        global_emb = checkpoint_global['global_emb']
        model.global_emb = global_emb
        if use_cuda:
            model.cuda()
            global_model.cuda()
        train_sub = '/train_history_sub.txt'
        train_ob = '/train_history_ob.txt'
        # if args.dataset == 'icews_know':
        #     valid_sub = '/test_history_sub.txt'
        #     valid_ob = '/test_history_ob.txt'
        # else:
        valid_sub = '/dev_history_sub.txt'
        valid_ob = '/dev_history_ob.txt'

        model_graph_file = "models/" + args.dataset + "/" + str(epoch) + "/rgcn_graph.pth"
        #model_graph_file = "./data/" + args.dataset + "/train_graphs.txt"
        print("loading " + model_graph_file)
        with open(model_graph_file, 'rb') as fp:
            graph_dict = pickle.load(fp)
        
        model.graph_dict = graph_dict
        # print(model)
        # print(global_model)

        with open('data/' + args.dataset+'/test_history_sub.txt', 'rb') as f:
            s_history_test_data = pickle.load(f)
        with open('data/' + args.dataset+'/test_history_ob.txt', 'rb') as f:
            o_history_test_data = pickle.load(f)

        s_history_test = s_history_test_data[0]
        s_history_test_t = s_history_test_data[1]
        o_history_test = o_history_test_data[0]
        o_history_test_t = o_history_test_data[1]
        
        with open('./data/' + args.dataset+train_sub, 'rb') as f:
            s_history_data = pickle.load(f)
        with open('./data/' + args.dataset+train_ob, 'rb') as f:
            o_history_data = pickle.load(f)
            
        with open('./data/' + args.dataset+valid_sub, 'rb') as f:
            s_history_valid_data = pickle.load(f)
        with open('./data/' + args.dataset+valid_ob, 'rb') as f:
            o_history_valid_data = pickle.load(f)
        valid_data = torch.from_numpy(valid_data)

        # print(s_history_valid_data[0])
        # print(s_history_valid_data[1])

        s_history = s_history_data[0]
        s_history_t = s_history_data[1]
        o_history = o_history_data[0]
        o_history_t = o_history_data[1]
        s_history_valid = s_history_valid_data[0]
        s_history_valid_t = s_history_valid_data[1]
        o_history_valid = o_history_valid_data[0]
        o_history_valid_t = o_history_valid_data[1]

        total_data = torch.from_numpy(total_data)
        if use_cuda:
            total_data = total_data.cuda()

        if epoch % args.valid_every == 0 or epoch == args.max_epochs:
            model.eval()
            global_model.eval()
            model.init_history(train_data, (s_history, s_history_t), (o_history, o_history_t), valid_data,
                           (s_history_valid, s_history_valid_t), (o_history_valid, o_history_valid_t), test_data,
                           (s_history_test, s_history_test_t), (o_history_test, o_history_test_t))
            model.latest_time = valid_data[0][3]
            total_loss = 0
            total_ranks = np.array([])

            for j in range(len(valid_data)):
                if j % 1000 == 0:
                    print("valid_data " + str(j) + "/" + str(len(valid_data)))

                batch_data = valid_data[j]
                s_hist = s_history_valid[j]
                o_hist = o_history_valid[j]
                s_hist_t = s_history_valid_t[j]
                o_hist_t = o_history_valid_t[j]

                if use_cuda:
                    batch_data = batch_data.cuda()

                with torch.no_grad():
                    ranks, loss = model.evaluate_filter(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t), global_model, total_data)
                    total_ranks = np.concatenate((total_ranks, ranks))
                    total_loss += loss.item()

            mrr = np.mean(1.0 / total_ranks)
            mr = np.mean(total_ranks)
            hits = []
            for hit in [1, 3, 10]:
                avg_count = np.mean((total_ranks <= hit))
                hits.append(avg_count)
                print("valid Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
            print("valid MRR (filtered): {:.6f}".format(mrr))
            print("valid MR (filtered): {:.6f}".format(mr))
            print("valid Loss: {:.6f}".format(total_loss / (len(valid_data))))

            if mrr > best_mrr:
                best_mrr = mrr
                best_state = epoch_model_state_file
                best_global2 = epoch_model_state_global_file2
                best_graph = model_graph_file

    print("validation done")
    print("Best model state: " + best_state)
    print("Best global2: " + best_global2)
    print("Best graph: " + best_graph)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18',
            help="dataset to use")
    parser.add_argument("--grad-norm", type=float, default=1.0,
    help="norm to clip gradient to")
    parser.add_argument("--max-epochs", type=int, default=20
                        ,
                        help="maximum epochs")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=1000,
                    help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument('--backup',	action='store_true')
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)

    args = parser.parse_args()
    print(args)
    train(args)

