import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import torch.nn as nn
import MultimodalSA.network as new_models
# import MultimodalSA.Jack_model as new_models
from utils import DiffLoss,SimilarityKL
from utils import to_gpu

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from config import get_config
train_config = get_config(mode='train')


checkpoint_path = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/AOTN_Normal/"
log_file = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/AOTN_Normal/MutimodalSALog.txt"
with open(log_file, "a") as file:
    file.write(f"Log File Time{train_config.name} in {train_config.data}\n") # 每次打开记录一下当前时间             

class Solver(object):
    def __init__(self, train_config, train_data_loader, dev_data_loader, test_data_loader,
                 is_train=True, model=None):
        
        self.loss_diff = DiffLoss()       
        self.loss_simi = SimilarityKL()
        self.loss_task = nn.MSELoss(reduction="mean") # MOSEI和MOSI情感分析属于回归任务,使用MSE损失

        self.train_config = train_config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

    def loss_function(self, y_tilde, y):
        """
        total_loss = w1 * cls_loss + w_2 * polar_loss + w_3 * scale_loss
        """
        loss_diff = self.get_diff_loss()
        loss_simi = self.get_simi_loss()
        loss_task = self.loss_task(y_tilde, y)

        loss = 0.5*loss_diff + 0.5*loss_simi + loss_task
        return loss

    def get_simi_loss(self,):

        # # losses between shared states CMD Original
        # loss = self.loss_simi(self.model.share_T, self.model.share_V, 5) # 使用CMD 需要加上 , 5
        # loss += self.loss_simi(self.model.share_T, self.model.share_A, 5)
        # loss += self.loss_simi(self.model.share_A, self.model.share_V, 5) 
        # loss = loss/3.0

        # SimilarityKL Redesign 
        input = (self.model.share_T, self.model.share_V, self.model.share_A)
        loss = self.loss_simi(input)
        return loss
    
    def get_diff_loss(self):

        shared_t = self.model.share_T
        shared_v = self.model.share_V
        shared_a = self.model.share_A
        private_t = self.model.private_t
        private_v = self.model.private_v
        private_a = self.model.private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss

    def train(self):
        #============================== 
        # Jack Add CheckPoint
        #==============================
        start_epoch = -1 # 手动指定从第几个epoch开始训练
        checkpoint_file = checkpoint_path + f"epoch{start_epoch}.pt"
        if checkpoint_path is not None and os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        print(f"=======Start training from epoch {start_epoch}==================")
        curr_patience = patience = self.train_config.patience # 6
        num_trials = self.train_config.trials # 3
 
        best_valid_loss = float('inf')

        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.train_config.milestones, gamma=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        train_losses = []
        train_acces = []
        for e in range(start_epoch,self.train_config.n_epoch): # 0-100

            # === 训练
            self.model.train()
            train_loss = []
            train_acc = []
            for batch in self.train_data_loader:
                y_tilde, y = self.model_input2output(batch)
                loss = self.loss_function(y_tilde, y)
                # acc = self.calc_metrics(y_tilde.cpu().detach().numpy(), y.cpu().detach().numpy(),mode="train") # MD 害群之马 pos/neg

                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                # train_acc.append(acc)

            train_losses.append(train_loss)
            # train_acces.append(train_acc)
            # print(f"Epoch {e} - Training acc: {round(np.mean(train_acc), 4)}")
            from datetime import datetime
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')            
            print(f"\nEpoch {e} - Training loss: {round(np.mean(train_loss), 4)}  Time: {time_now}\n")
            with open(log_file, "a") as file:
                # file.write(f"Epoch {e} - Training acc: {round(np.mean(train_acc), 4)} Time: {time_now}\n")            
                file.write(f"\nEpoch {e} - Training loss: {round(np.mean(train_loss), 4)}  Time: {time_now}\n")

            # ==== 验证
            valid_loss, valid_acc = self.eval(mode="dev")
            print(f"val loss: {round(np.mean(valid_loss), 4)}, val acc: {round(np.mean(valid_acc), 4)}")
            with open(log_file, "a") as file:
                file.write(f"val loss: {round(np.mean(valid_loss), 4)}, val acc: {round(np.mean(valid_acc), 4)}\n")            
            
            # ==== 测试
            if e % self.train_config.test_duration == 0: # 1 个 epoch 测试一次
                # lr_scheduler.step()
                test_loss, test_acc = self.eval(e,mode="test")
                print(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}")
                with open(log_file, "a") as file:
                    file.write(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}\n")            

            # ===== 交叉验证(正则化): 若模型一直没有提升，则只允许更新 6*3=18 epoch self.train_config.patience * self.train_config.trials
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                with open(log_file, "a") as file:
                    file.write(f"Found new best model on dev set!\n")            
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'{checkpoint_path}/devBest/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'{checkpoint_path}/devBest/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    with open(log_file, "a") as file:
                        file.write(f"Running out of patience, loading previous best model.")  
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'{checkpoint_path}devBest/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'{checkpoint_path}devBest/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
                    with open(log_file, "a") as file:
                        file.write(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")              
            file.close()

            # ====== 定点保存
            if e % 10 == 0: # 10 个 epoch 保存一次
                self.model.eval() # 将模型设置为评估模式，这样可以确保所有的参数都被正确保存 0524
                torch.save({
                    'epoch': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },checkpoint_path + f"epoch{e}.pt")
                test_loss, test_acc = self.eval(e,mode="test", to_print=True)
            
            print("#" * 100)
            with open(log_file, "a") as file:
                file.write("#" * 100)       

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break            

        # 最终测试
        self.eval(self.train_config.n_epoch-10,mode="test", to_print=True)

    def eval(self, e=None, mode=None, to_print=False):
        assert (mode is not None)
        self.model.eval()

        y_true, y_pred, y2_pred = [], [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader
            if to_print:
                self.model.load_state_dict(torch.load(
                    f'{checkpoint_path}devBest/model_{self.train_config.name}.std'))
                print("================== Start Testing ====================================")

        with torch.no_grad():
            for batch in dataloader:
                y_tilde, y = self.model_input2output(batch)
                loss = self.loss_function(y_tilde, y)

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())

                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False): # Jack Change 反了
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

        # ============ pos - neg
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)

        if mode=="train":
            pass
        if mode=="dev":
            print("val_mult_acc7: ", mult_a7)
            with open(log_file, "a") as file:
                    file.write(f"\nval_mult_acc7 {mult_a7}\n")
        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
        if mode=="test":
            print("\ntest_mult_acc7: ", mult_a7)
            print("mae: ", mae)
            print("corr: ", corr)
            print("F1 (pos/neg) :", f_score)
            with open(log_file, "a") as file:
                    file.write(f"\ntest_mult_acc7 {mult_a7}\n")
                    file.write(f"mae: {mae}\n")
                    file.write(f"corr: {corr}\n")
                    file.write(f"F1 (pos/neg) : {f_score}\n")
        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

        if to_print:
            print("mae: ", mae)
            print("corr: ", corr)
            print("mult_acc5: ", mult_a5)
            print("mult_acc7: \n", mult_a7)
            print("Classification Report (pos/neg) \n:")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("\nAccuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            print("F1 (pos/neg) :", f_score)
            with open(log_file, "a") as file:
                file.write(f"mae: {mae}\n")
                file.write(f"corr: {corr}\n")
                file.write(f"mult_acc5: {mult_a5}")
                file.write(f"mult_acc7: {mult_a7}\n")
                file.write(f"\nClassification Report (pos/neg) : \n")
                file.write(classification_report(binary_truth, binary_preds, digits=5))
                file.write(f"\nAccuracy (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")
                file.write(f"F1 (pos/neg) : {f_score}\n")

        if mode=="train": # Jack Add 0531 pos/neg 也要提上日程
            pass
        if mode=="dev":
            print("Accuracy_val (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            with open(log_file, "a") as file:
                    file.write(f"Accuracy_val (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")        
        if mode=="test":
            print("test_Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            with open(log_file, "a") as file:
                    file.write(f"Accuracy (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")

        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        f_score2 = f1_score((test_preds >= 0), (test_truth >= 0), average='weighted')

        if to_print:
            print("Classification Report (non-neg/neg) :\n")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("\nAccuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            print("F1 (non-neg/neg) :", f_score2)
            with open(log_file, "a") as file:
                file.write(f"Classification Report (non-neg/neg): \n")
                file.write(classification_report(binary_truth, binary_preds, digits=5))
                file.write(f"\nAccuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")
                file.write(f"F1 (non-neg/neg) : {f_score2}\n")
        if mode=="train":
            pass
        if mode=="dev":
            print("Accuracy_val (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            with open(log_file, "a") as file:
                    file.write(f"Accuracy_val (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")        
        if mode=="test":
            print("F1 (non-neg/neg) :", f_score2)            
            print("test_Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            with open(log_file, "a") as file:
                    file.write(f"F1 (non-neg/neg) : {f_score2}\n")
                    file.write(f"test_Accuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")

        return accuracy_score(binary_truth, binary_preds) # 试试 0524 accuracy = (pred.eq(y)).float().mean() # 返回的是 none-neg/neg

    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(new_models, self.train_config.model)(self.train_config)  # init the model

        # Final list
        for name, param in self.model.named_parameters():
            # Bert freezing customizations
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            if self.train_config.data == "ur_funny": # Jack Add
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            # elif self.train_config.data == "ur_funny": # Jack Change 0523 
            #     if "bert" in name:
            #         param.requires_grad = False
            if self.train_config.data == "mosi": # Jack Change 0523 
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False       
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    def model_input2output(self, batch):
        """
        get output from model input
        :param batch: batch
        :return: y_tilde: model predict output
                 y: true label
        """
        self.model.zero_grad()

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