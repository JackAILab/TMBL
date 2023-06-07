import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import torch.nn as nn
import new_models
from utils import corr_loss
from utils import cos_loss
from utils import to_gpu

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

class Solver(object):
    def __init__(self, train_config, train_data_loader, dev_data_loader, test_data_loader,
                 is_train=True, model=None):
        self.scale_criterion = corr_loss()
        self.polar_criterion = cos_loss()
        self.criterion = nn.MSELoss(reduction="mean")
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
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
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

        v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

        v = to_gpu(v)
        a = to_gpu(a)
        y = to_gpu(y)

        bert_sent = to_gpu(bert_sent)
        bert_sent_type = to_gpu(bert_sent_type)
        bert_sent_mask = to_gpu(bert_sent_mask)

        y_tilde = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

        return y_tilde, y

    def loss_function(self, y_tilde, y):
        """
        total_loss = w1 * cls_loss + w_2 * polar_loss + w_3 * scale_loss
        """
        polar_loss = self.polar_criterion(self.model.polar_vector, y, y_tilde)
        scale_loss = self.scale_criterion(self.model.scale, y)
        cls_loss = self.criterion(y_tilde, y)

        loss = self.train_config.cls_weight * cls_loss + self.train_config.polar_weight * polar_loss + self.train_config.scale_weight * scale_loss
        return loss

    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = self.train_config.trials

        best_valid_loss = float('inf')

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        train_losses = []
        for e in range(self.train_config.n_epoch):
            self.model.train()
            train_loss = []
            for batch in self.train_data_loader:
                y_tilde, y = self.model_input2output(batch)
                loss = self.loss_function(y_tilde, y)

                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_losses.append(train_loss)
            print(f"Epoch {e} - Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc = self.eval(mode="dev")
            print(f"val loss: {round(np.mean(valid_loss), 4)}, val acc: {round(np.mean(valid_acc), 4)}")

            if e % self.train_config.test_duration == 0:
                print("test dataset:")
                test_loss, test_acc = self.eval(mode="test")
                print(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}")

            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/PXMixer/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/PXMixer/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/PXMixer/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/PXMixer/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            print("#" * 100)

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break
        self.eval(mode="test", to_print=True)

    def eval(self, mode=None, to_print=False):
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
                    f'checkpoints/PXMixer/model_{self.train_config.name}.std'))

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

        print("mult_acc7: ", mult_a7)

        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

        # pos - neg
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)

        if to_print:
            print("mae: ", mae)
            print("corr: ", corr)
            print("mult_acc5: ", mult_a5)
            print("Classification Report (pos/neg) :")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            print("F1:", f_score)

        print("Accuracy (pos/neg)  ", accuracy_score(binary_truth, binary_preds))
    
        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        f_score2 = f1_score((test_preds >= 0), (test_truth >= 0), average='weighted')

        if to_print:
            print("Classification Report (non-neg/neg) :")
            print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            print("F1:", f_score2)

        print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
        return accuracy_score(binary_truth, binary_preds)
