import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import MultimodalSA.Jack_model_AOTrans_CLIP as new_models
from data_loader import get_loader
from config import get_config
from torch import optim

train_config = get_config(mode='test')
train_config.visual_size = 35
train_config.acoustic_size = 74

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

checkpoint_path = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/TransformerPX/"
log_file = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/TransformerPX/MutimodalSALog.txt"
with open(log_file, "a") as file:
    file.write(f"Log File Time{train_config.name} in {train_config.data}\n") # 每次打开记录一下当前时间    

model =  getattr(new_models, train_config.model)(train_config)
optimizer = optim.Adam

def to_gpu(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
    return x

def model_input2output(self, batch):
    """
    get output from model input
    :param batch: batch
    :return: y_tilde: model predict output
                y: true label
    """
    model.zero_grad()

    v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch # ([63, 64, 35]) ([63, 64, 74]) ([64, 1]) 64 ([64, 65]) 
    # ([42, 128, 47]) ([42, 128, 74]) ([128, 1]) 128 ([128, 44])
    v = to_gpu(v)
    a = to_gpu(a)
    y = to_gpu(y)

    bert_sent = to_gpu(bert_sent)
    bert_sent_type = to_gpu(bert_sent_type)
    bert_sent_mask = to_gpu(bert_sent_mask)

    y_tilde = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

    return y_tilde, y # [32,1] [32,1]

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy 计算多类精度 w.r.t. groundtruth 
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3, a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3, a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)


    print("\ntest_mult_acc7: ", mult_a7)
    print("mae: ", mae)
    print("corr: ", corr)
    print("F1:", f_score)
    with open(log_file, "a") as file:
            file.write(f"\ntest_mult_acc7 {mult_a7}\n")
            file.write(f"mae: {mae}\n")
            file.write(f"corr: {corr}\n")
            file.write(f"F1: {f_score}\n")

    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

    # pos - neg
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    print("mae: ", mae)
    print("corr: ", corr)
    print("mult_acc5: ", mult_a5)
    print("mult_acc7: \n", mult_a7)
    print("Classification Report (pos/neg) \n:")
    print(classification_report(binary_truth, binary_preds, digits=5))
    print("\nAccuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
    print("F1:", f_score)
    with open(log_file, "a") as file:
        file.write(f"mae: {mae}\n")
        file.write(f"corr: {corr}\n")
        file.write(f"mult_acc5: {mult_a5}")
        file.write(f"mult_acc7: {mult_a7}\n")
        file.write(f"\nClassification Report (pos/neg): \n")
        file.write(classification_report(binary_truth, binary_preds, digits=5))
        file.write(f"\nAccuracy (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")
        file.write(f"F1: {f_score}\n")
    
    # non-neg - neg
    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)

    f_score2 = f1_score((test_preds >= 0), (test_truth >= 0), average='weighted')

    print("Classification Report (non-neg/neg) :\n")
    print(classification_report(binary_truth, binary_preds, digits=5))
    print("\nAccuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
    print("F1:", f_score2)
    with open(log_file, "a") as file:
        file.write(f"Classification Report (non-neg/neg): \n")
        file.write(classification_report(binary_truth, binary_preds, digits=5))
        file.write(f"\nAccuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")
        file.write(f"F1: {f_score2}\n")
    
    print("test_Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
    with open(log_file, "a") as file:
            file.write(f"test_Accuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")
    return accuracy_score(binary_truth, binary_preds)

def eval(e=None, mode=None):

    model.eval() # 准备测试

    test_config = get_config(mode='test')
    test_data_loader = get_loader(test_config, shuffle = False)

    dataloader = test_data_loader

    checkpoint_file = checkpoint_path + f"epoch{e}.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("================== Start Testing ====================================")

    with torch.no_grad():
        for batch in dataloader:
            
            y_tilde, y = model_input2output(batch)

            y_pred.append(y_tilde.detach().cpu().numpy())

            y_true.append(y.detach().cpu().numpy())


    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred = np.concatenate(y_pred, axis=0).squeeze()

    accuracy = calc_metrics(y_true, y_pred, mode)

    return accuracy

test_loss, test_acc = eval(e=290,mode="test")
print(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}")
with open(log_file, "a") as file:
    file.write(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}\n")       


    