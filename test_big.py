import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from MVSOPPIS_model_bigmodel import *
from tqdm import tqdm


# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Log/test_bigmodel/model/"

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
    # y1 = torch.abs(y1)
    # y2 = torch.abs(y2)
    # y3 = torch.abs(y3)
    return y1.reshape(-1), y2.reshape(-1), y3.reshape(-1)

def com_acc(x, y):
    cnt1 = {}
    cnt2 = {}
    for item in zip(x, y):
        a = int(item[0])
        b = int(item[1])
        if cnt1.get(a) == None:
            cnt1[a] = 1
        else:
            cnt1[a] += 1

        if cnt2.get(b) == None and a == b:
            cnt2[b] = 1
        if cnt2.get(b) != None and a == b:
            cnt2[b] += 1

        if cnt2.get(a) == None:
            cnt2[a] = 0

    cnt = {}
    for key in cnt1.keys():
        cnt[key] = cnt2[key] / cnt1[key]
    return cnt
        


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []

    best_name = None
    best_pro = None
    best_true = None
    best_acc = 0
    

    pred_dict = {}

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(device)
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

            best_threshold = 0.15
            binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred[:, 1]]
            binary_true = y_true
            binary_acc = metrics.accuracy_score(binary_true, binary_pred)
            pred_01,_,_ = make_edge(torch.tensor(binary_pred))
            # pred_010,_,_ = make_point(torch.tensor(binary_pred))
            # true_010,_,_ = make_point(torch.tensor(binary_true))
            true_01,_,_ = make_edge(torch.tensor(binary_true))
            res = com_acc(true_01, pred_01)
            acc = res[1] + res[-1]
            
            # print("edge: ", com_acc(true_01, pred_01))
            # print("point: ", com_acc(true_010, pred_010))

            if acc > best_acc:
                best_acc = acc
                best_name = sequence_names
                best_pro = binary_pred
                best_true = binary_true



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

    pred_01,_,_ = make_edge(torch.tensor(binary_pred))
    pred_010,_,_ = make_point(torch.tensor(binary_pred))
    true_010,_,_ = make_point(torch.tensor(binary_true))
    true_01,_,_ = make_edge(torch.tensor(binary_true))
    # pred_01 = pred_01.cpu().detach().numpy()
    # pred_010 = pred_010.cpu().detach().numpy()
    # true_01 = true_01.cpu().detach().numpy()
    # true_010 = true_010.cpu().detach().numpy()
    # print("edge: ", com_acc(true_01, pred_01))
    # print("point: ", com_acc(true_010, pred_010))
    
    

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred, drop_intermediate=False)
    mask = ~np.isnan(precisions) & ~np.isnan(recalls)
    precisions = precisions[mask]
    recalls = recalls[mask]
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


def test(test_dataframe, psepos_path):
    embed_dirs = "./Embeddingsss/"
    feature_Path = "./nFeature/"
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path, embed_dir=embed_dirs, Feature_path=feature_Path), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)

    # for model_name in sorted(os.listdir(Model_Path)):
    if True:
        # model_name = "Full_model54.pkl"
        model_name = "best_model.pkl"
        print(model_name)
        model = MVSOPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)


        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])


def test_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe, psepos_path)


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)

    with open("./nFeature/PP-250_Test.pkl", "rb") as f:
        Test_250 = pickle.load(f)

    Btest_31_6 = {}
    with open(Dataset_Path + "bound_unbound_mapping31-6.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31_6[bound_ID] = Test_60[bound_ID]

    Test60_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    Test250_psepos_Path = './Datasets/customed_data/PP/PP_psepos_SC.pkl'
    Test315_28_psepos_Path = './Feature/psepos/Test315-28_psepos_SC.pkl'
    Btest31_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    UBtest31_28_psepos_Path = './Feature/psepos/UBtest31-6_psepos_SC.pkl'

    # print("Evaluate GraphPPIS on Test_60")
    # test_one_dataset(Test_60, Test60_psepos_Path)

    print("Evaluate GraphPPIS on Test_250")
    test_one_dataset(Test_250, Test250_psepos_Path)

    # print("Evaluate GraphPPIS on Test_315-28")
    # test_one_dataset(Test_315_28, Test315_28_psepos_Path)

    # print("Evaluate GraphPPIS on Btest_31-6")
    # test_one_dataset(Btest_31_6, Btest31_psepos_Path)

    # print("Evaluate GraphPPIS on UBtest_31-6")
    # test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path)


if __name__ == "__main__":
    main()


