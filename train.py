import sys
import time

import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from MVSOPPIS_model import *
# import dgl


# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
Log_path = "./Log/"
model_time = None


def make_edge(x):
    shape2 = (1, 1, x.shape[0])
    kernel1 = torch.tensor([[[-1,1]]], dtype=x.dtype, device=x.device)
    kernel2 = torch.tensor([[[-1,0,1]]], dtype=x.dtype, device=x.device)
    kernel3 = torch.tensor([[[1,0,-1]]], dtype=x.dtype, device=x.device)
    y1 = F.conv1d(x.reshape(shape2), kernel1)
    y2 = F.conv1d(x.reshape(shape2), kernel2)
    y3 = F.conv1d(x.reshape(shape2), kernel3)
    # y1 = torch.abs(y1)
    # y2 = torch.abs(y2)
    # y3 = torch.abs(y3)
    return y1.reshape(-1), y2.reshape(-1), y3.reshape(-1)

def make_point(x):
    shape2 = (1, 1, x.shape[0])
    kernel1 = torch.tensor([[[-1,2,-1]]], dtype=x.dtype, device=x.device)
    kernel2 = torch.tensor([[[-1,-1,4,-1,-1]]], dtype=x.dtype, device=x.device)
    kernel3 = torch.tensor([[[-1,-2,6,-2,-1]]], dtype=x.dtype, device=x.device)
    y1 = F.conv1d(x.reshape(shape2), kernel1)
    y2 = F.conv1d(x.reshape(shape2), kernel2)
    y3 = F.conv1d(x.reshape(shape2), kernel3)
    
    return y1.reshape(-1), y2.reshape(-1), y3.reshape(-1)

def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        seq_name, _, labels, node_features, G_batch, adj_matrix = data

        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda().float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            G_batch = G_batch.to(torch.device('cuda:0'))
            adj_matrix = Variable(adj_matrix.cuda())
            y_true = Variable(labels.cuda())
            # print(node_features.shape)
        else:
            node_features = Variable(node_features.float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            adj_matrix = Variable(adj_matrix)
            y_true = Variable(labels)

        adj_matrix = torch.squeeze(adj_matrix)
        y_true = torch.squeeze(y_true)
        y_true = y_true.long()

        y_pred = model(node_features, G_batch, adj_matrix, seq_name)


        #####################################################
        softmax = torch.nn.Softmax(dim=1)
        y_predd = softmax(y_pred)
        a1, a2, a3 = make_edge(y_predd[:,1])
        b1, b2, b3 = make_edge(y_true.to(a1.dtype))
        a11, a22, a33 = make_point(y_predd[:,1])
        b11, b22, b33 = make_point(y_true.to(a1.dtype))
        #####################################################

        # calculate loss
        
        a = model.criterion(y_pred, y_true)
        b = 0.5*(F.mse_loss(a1, b1) + F.mse_loss(a2, b2) + F.mse_loss(a3, b3)) +  0.5*(F.mse_loss(a11, b11) + F.mse_loss(a22, b22) + F.mse_loss(a33, b33))
        
        loss = a + 0.2*b

        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                y_true = Variable(labels.cuda())

            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            y_pred = model(node_features, G_batch, adj_matrix, sequence_names)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def train(model, train_dataframe, valid_dataframe, fold = 0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)
    writer = SummaryWriter(log_dir='./logs'  + str(fold))

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        # 记录训练损失
        writer.add_scalar('Train Loss', epoch_loss_train_avg, epoch)
        # 记录其他训练指标
        writer.add_scalar('Train Binary Accuracy', result_train['binary_acc'], epoch)
        writer.add_scalar('Train AUC', result_train['AUC'], epoch)
        writer.add_scalar('Train AUPRC', result_train['AUPRC'], epoch)

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred, 0.5)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])

        # 记录验证损失
        writer.add_scalar('Validation Loss', epoch_loss_valid_avg, epoch)
        # 记录其他验证指标
        writer.add_scalar('Validation Binary Accuracy', result_valid['binary_acc'], epoch)
        writer.add_scalar('Validation Precision', result_valid['precision'], epoch)
        writer.add_scalar('Validation Recall', result_valid['recall'], epoch)
        writer.add_scalar('Validation F1', result_valid['f1'], epoch)
        writer.add_scalar('Validation AUC', result_valid['AUC'], epoch)
        writer.add_scalar('Validation AUPRC', result_valid['AUPRC'], epoch)
        writer.add_scalar('Validation MCC', result_valid['mcc'], epoch)

        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

        model.scheduler.step(result_valid['AUPRC'])

    return best_epoch, best_val_auc, best_val_aupr


def cross_validation(all_dataframe, fold_number=5):
    print("Random seed:", SEED)
    print("The base Model type:", BASE_MODEL_TYPE)
    print("Add node features:", ADD_NODEFEATS)
    print("Map cutoff:", MAP_CUTOFF)
    print("Use edge features or not while using GAT model:", USE_EFEATS)
    print("The parameter of normalizing the distance:", DIST_NORM)
    print("Feature dim:", INPUT_DIM)
    print("Hidden dim:", HIDDEN_DIM)
    print("Layer:", LAYER)
    print("Dropout:", DROPOUT)
    print("Alpha:", ALPHA)
    print("Lambda:", LAMBDA)
    print("Learning rate:", LEARNING_RATE)
    print("Training epochs:", NUMBER_EPOCHS)
    print()

    # 取出dataframe中的值
    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")

        model = MVSOPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1


    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


def train_full_model(all_dataframe, aver_epoch):
    writer = SummaryWriter(log_dir='./logs')
    best_auprc = 0

    print("\n\nTraining a full model using all training data...\n")
    model = MVSOPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
    if torch.cuda.is_available():
        model.cuda()

    train_loader = DataLoader(dataset=ProDataset(all_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        # 记录训练损失
        writer.add_scalar('Train Loss', epoch_loss_train_avg, epoch)

        # 记录其他训练指标
        writer.add_scalar('Train Binary Accuracy', result_train['binary_acc'], epoch)
        writer.add_scalar('Train AUC', result_train['AUC'], epoch)
        writer.add_scalar('Train AUPRC', result_train['AUPRC'], epoch)

        if epoch + 1 in [aver_epoch, 45]:
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))  # 保存模型的参数
        



class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass



def main():
    if not os.path.exists(Log_path): os.makedirs(Log_path)

    with open(Dataset_Path + "Train_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA')  # remove the protein with error sequence in the train dataset

    IDs, sequences, labels = [], [], []

    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_dataframe = pd.DataFrame(train_dic)
    aver_epoch = cross_validation(train_dataframe, fold_number=5)

    train_full_model(train_dataframe, aver_epoch)
    
import random
def setup_seed(seed):
    np.random.seed(seed) # numpy 的设置
    random.seed(seed)  # python random module
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了使得hash随机化，使得实验可以复现
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(seed) # 为cpu设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU为，所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False # 设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
        torch.backends.cudnn.deterministic = True # 每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
                                                    # 应该可以保证每次运行网络的时候相同输入的输出是固定的

if __name__ == "__main__":
    
    if model_time is not None:
        checkpoint_path = os.path.normpath(Log_path +"/"+ model_time)
    else:
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # checkpoint_path = os.path.normpath(Log_path + "/" + localtime)
        checkpoint_path = os.path.normpath(Log_path + "/test" )

        os.makedirs(checkpoint_path)
    Model_Path = os.path.normpath(checkpoint_path + '/model')
    if not os.path.exists(Model_Path): os.makedirs(Model_Path)

    sys.stdout = Logger(os.path.normpath(checkpoint_path + '/training.log'))
    main()
    sys.stdout.log.close()

