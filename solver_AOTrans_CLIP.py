import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import torch.nn as nn
# import new_models
# import MultimodalSA.Jack_model_AOTrans as new_models
# import MultimodalSA.Jack_model_AOTrans_CLIP as new_models # 这个版本py可以对UR_FUNNY数据处理，但是90%的acc存在问题
import MultimodalSA.Jack_model_AOTrans_CLIP_NOCom as new_models
from utils import corr_loss
from utils import cos_loss
from utils import CosineSimilarity # JackAdd 0521
from utils import to_gpu

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from config import get_config
train_config = get_config(mode='train')

# checkpoint_path = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/PXMixer"
# log_file = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/PXMixer/MSALog_PXMixer.txt"
checkpoint_path = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/TransformerPX/"
log_file = "/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/checkpoints/TransformerPX/MutimodalSALog.txt"
with open(log_file, "a") as file:
    file.write(f"Log File Time{train_config.name} in {train_config.data}\n") # 每次打开记录一下当前时间             

class Solver(object):
    def __init__(self, train_config, train_data_loader, dev_data_loader, test_data_loader,
                 is_train=True, model=None):
        self.scale_criterion = corr_loss()
        self.polar_criterion = cos_loss()
        self.cosine_criterion = CosineSimilarity() # JackAdd 0521
        self.criterion = nn.MSELoss(reduction="mean") # MOSEI和MOSI情感分析属于回归任务,使用MSE损失
        self.ce_criterion = nn.CrossEntropyLoss() # JackAdd 0524 UR_FUNNY 需要使用交叉熵损失,属于二分类任务
        self.train_config = train_config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

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

    def loss_function(self, y_tilde, y):
        """
        total_loss = w1 * cls_loss + w_2 * polar_loss + w_3 * scale_loss
        """
        if self.train_config.data is not "ur_funny":
            polar_loss = self.polar_criterion(self.model.polar_vector, y, y_tilde)
            scale_loss = self.scale_criterion(self.model.scale, y)
            cls_loss = self.criterion(y_tilde, y)
            transformer_polar_loss = self.polar_criterion(self.model.multimodal, y, y_tilde)
            transformer_cosine_loss = self.cosine_criterion(self.model.utterance_t,self.model.utterance_v)

            # loss = self.train_config.cls_weight * cls_loss + self.train_config.polar_weight * polar_loss + self.train_config.scale_weight * scale_loss # original loss
            loss = self.train_config.cls_weight * cls_loss + \
            self.train_config.polar_weight * polar_loss + self.train_config.scale_weight * scale_loss + \
            2*self.train_config.polar_weight * transformer_polar_loss + \
            2*self.train_config.cls_weight * transformer_cosine_loss # JackAdd 0521 
            
            # polar_loss = self.polar_criterion(self.model.polar_vector, y, y_tilde)
            # scale_loss = self.scale_criterion(self.model.scale, y)
            # cls_loss = self.criterion(y_tilde, y)

            # loss = self.train_config.cls_weight * cls_loss + \
            # self.train_config.polar_weight * polar_loss + self.train_config.scale_weight * scale_loss

        if self.train_config.data is "ur_funny":
            ce_loss = self.ce_criterion(y_tilde, y)
            loss = ce_loss
            # transformer_cosine_loss = self.cosine_criterion(self.model.utterance_t,self.model.utterance_v)
            # loss = ce_loss + transformer_cosine_loss
            pass

        return loss

    def train(self):
        #============================== 
        # Jack Add CheckPoint
        #==============================
        start_epoch = 0 # 手动指定从第几个epoch开始训练
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
                acc = self.calc_metrics(y_tilde.cpu().detach().numpy(), y.cpu().detach().numpy(),mode="train")

                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                train_acc.append(acc)
            train_losses.append(train_loss)
            train_acces.append(train_acc)
            print(f"Epoch {e} - Training acc: {round(np.mean(train_acc), 4)}\n")
            print(f"Epoch {e} - Training loss: {round(np.mean(train_loss), 4)}\n")
            from datetime import datetime
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            with open(log_file, "a") as file:
                file.write(f"\n Epoch {e} - Training acc: {round(np.mean(train_acc), 4)} Time: {time_now}")            
                file.write(f"\n Epoch {e} - Training loss: {round(np.mean(train_loss), 4)}\n")

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
            
            # lr_scheduler.step()
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
                checkpoint_file = checkpoint_path + f"epoch{e}.pt"
                checkpoint = torch.load(checkpoint_file)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

    def calc_metrics(self, y_pred, y_true, mode=None, to_print=False): # Jack Change 反了
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

        if to_print:
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

        if to_print:
            print("Classification Report (non-neg/neg) :\n")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("\nAccuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            print("F1:", f_score2)
            with open(log_file, "a") as file:
                file.write(f"Classification Report (non-neg/neg): \n")
                file.write(classification_report(binary_truth, binary_preds, digits=5))
                file.write(f"\nAccuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")
                file.write(f"F1: {f_score2}\n")
        if mode=="train":
            pass
        if mode=="dev":
            print("val_Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            with open(log_file, "a") as file:
                    file.write(f"val_Accuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")        
        if mode=="test":
            print("test_Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            with open(log_file, "a") as file:
                    file.write(f"test_Accuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")
        return accuracy_score(binary_truth, binary_preds) # 试试 0524 accuracy = (pred.eq(y)).float().mean()

