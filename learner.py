import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from module.loss import GeneralizedCELoss
from module.resnets_vision import dic_models
from data.util import get_dataset, IdxDataset
from torch.utils.data import DataLoader
from util import EMA
from module.utils import dic_functions, mixup_data
from config import *
from tqdm import tqdm

set_seed = dic_functions['set_seed']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
write_to_file = dic_functions['write_to_file']

class trainer():

    def __init__(self, args):
        self.run_type = args.run_type
        self.loader_config = dataloader_confg[args.dataset_in]
        self.loss_contr = args.loss_contr
        self.data_og = args.dataset_in

        self.mix_up_val = args.mix_up_val
        self.thresh = args.thresh
        if args.dataset_in == 'CMNIST':
            print("[DATASET][CMNIST]")
            self.dataset_in = 'ColoredMNIST-Skewed'+str(args.bias_ratio_og)+'-Severity4'
        else:
            print("Invalid Dataset")
            exit()

        self.model_in_d = args.model_in_d
        self.model_in_b = args.model_in_b
        self.preprocess = args.preprocess
        self.epoch_preprocess = args.epoch_preprocess
        self.train_samples = args.train_samples
        self.bias_ratio = args.bias_ratio
        self.reduce = args.reduce
        self.target_attr_idx = 0
        self.bias_attr_idx = 1
        self.num_classes = 10
        
        self.info_str = "---------------------\n"
        self.info_str += "TimeStamp: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.info_str += "Original Dataset: {}\n".format(self.dataset_in)
        self.info_str += "ThreshMixUp: {}\n".format(self.thresh)
        self.info_str += "Run Type: {}\n".format(self.run_type)
        self.info_str += "Preprocess: {}\n".format(self.preprocess)
        self.info_str += "Type :{}\n".format(args.type)
        self.info_str += "Model D: {}\n".format(self.model_in_d)
        self.info_str += "Model B: {}\n".format(self.model_in_b)
        self.info_str += "Train Samples: {}\n".format(self.train_samples)
        self.info_str += "Bias Conflicting Samples: {}\n".format(int(self.bias_ratio * self.train_samples))
        self.info_str += "Bias Ratio: {}\n".format(self.bias_ratio)
        self.info_str += "Reduce: {}\n".format(self.reduce)
        file_name_ = list(map(str, [self.data_og, self.run_type, str(self.train_samples), str(self.bias_ratio), self.reduce, self.preprocess, self.thresh, self.mix_up_val, self.loss_contr]))
        names_attr = ['D','R','TS','BR','Re','P','T', 'M', 'LC']

        file_name = ''
        for i in range(len(file_name_)):
            file_name += names_attr[i] + '_'+file_name_[i] + '_'
        
        file_name = file_name[:-1]
     
        self.name_run = file_name
        file_name += '.txt'
        write_to_file('results_text/' + file_name, self.info_str)


    def rotation_ssl_data(self, images, labels):
        labels = torch.zeros(len(labels))
        images_90 = TF.rotate(images, 90)
        labels_90 = torch.ones(len(labels))
        images_180 = TF.rotate(images, 180)
        labels_180 = torch.ones(len(labels))*2
        images_270 = TF.rotate(images, 270)
        labels_270 = torch.ones(len(labels))*3
        images = torch.cat((images, images_90, images_180, images_270), dim=0)
        labels = torch.cat((labels, labels_90, labels_180, labels_270), dim=0) 

        return images, labels

    def preprocess_model(self, model, dataloader ,epochs = 100, preprocess = 'none'):
        if preprocess == 'none':
            return model
        else:
            mlp = nn.Sequential(
                nn.Linear(model.fc.in_features, 4),
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)

            for epoch in range(1, epochs+1):
                for ix, (index,data,attr) in enumerate(dataloader):
                    
                    label = attr[:, self.target_attr_idx]
                    data_rot, label_rot = self.rotation_ssl_data(data, label)

                    data_rot = data_rot.to(device)
                    label_rot = label_rot.to(device)

                    if torch.cuda.device_count() == 1:
                        z_r = model.features(data_rot)
                    else:
                        z_r = model.module.features(data_rot)

                    z_r = mlp(z_r)

                    loss_rotate = self.criterion(z_r, label_rot.long()).mean()

                    optimizer.zero_grad()
                    optimizer_mlp.zero_grad()
                    loss_rotate.backward()
                    optimizer.step()
                    optimizer_mlp.step()
                
                print('Progress [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss_rotate.item()))
            return model

    def store_results(self, test_accuracy, test_accuracy_epoch, test_cheat):
        write_to_file('results_text/'+ self.name_run +'.txt','[Final Epoch Train Accuracy]'+str(test_accuracy)+"[Final Epoch Test Accuracy]"+str(test_accuracy_epoch)+ '[Test Accuracy]'+str(test_cheat))

    def datasets(self):
        self.train_dataset = get_dataset(
            self.dataset_in,
            dataset_split="train",
            transform_split="train",)
        self.test_dataset = get_dataset(
            self.dataset_in,
            dataset_split="eval",
            transform_split="eval",)
        self.valid_dataset = get_dataset(
            self.dataset_in,
            dataset_split="train",
            transform_split="train",
        )

    def reduce_data(self):
        indices_train_biased = self.train_dataset.attr[:,self.target_attr_idx] == self.train_dataset.attr[:,self.bias_attr_idx]
        indices_train_biased = indices_train_biased.nonzero().squeeze()

        nums_train_biased = []
        for i in range(self.num_classes):
            nums_train_biased.append(np.random.choice(indices_train_biased[self.train_dataset.attr[indices_train_biased,self.target_attr_idx] == i], int((1-self.bias_ratio) * self.train_samples/self.num_classes) , replace=False))
        nums_train_biased = np.concatenate(nums_train_biased)


        indices_train_unbiased = self.train_dataset.attr[:,self.target_attr_idx] != self.train_dataset.attr[:,self.bias_attr_idx]
        indices_train_unbiased = indices_train_unbiased.nonzero().squeeze()
        
        nums_train_unbiased = []
        for i in range(self.num_classes):
            nums_train_unbiased.append(np.random.choice(indices_train_unbiased[self.train_dataset.attr[indices_train_unbiased,self.target_attr_idx] == i], int(self.bias_ratio * self.train_samples/self.num_classes) , replace=False))
        
        nums_train_unbiased = np.concatenate(nums_train_unbiased)

        nums_train = np.concatenate((nums_train_biased, nums_train_unbiased))

        nums_valid_unbiased = []
        while len(nums_valid_unbiased) < 1000:
            i = np.random.randint(0, len(self.valid_dataset))
            if self.valid_dataset.attr[i,self.target_attr_idx] != self.valid_dataset.attr[i,self.bias_attr_idx] and i not in nums_train:
                nums_valid_unbiased.append(i)
        nums_valid_unbiased = np.array(nums_valid_unbiased)

        self.valid_dataset.attr = self.valid_dataset.attr[nums_valid_unbiased]
        self.valid_dataset.data = self.valid_dataset.data[nums_valid_unbiased]
        self.valid_dataset.__len__ = 1000
        self.valid_dataset.query_attr = self.valid_dataset.attr[:, torch.arange(2)]
        
        self.train_dataset.attr = self.train_dataset.attr[nums_train]
        self.train_dataset.data = self.train_dataset.data[nums_train]
        self.train_dataset.__len__ = self.train_samples
        self.train_dataset.query_attr = self.train_dataset.attr[:, torch.arange(2)]
        del indices_train_biased, indices_train_unbiased, nums_train_biased, nums_train_unbiased, nums_train, nums_valid_unbiased

    def dataloaders(self):

        print("[Size of the Dataset]["+str(len(self.train_dataset))+"]")
        print("[Conflicting Samples in Training Data]["+str(len(self.train_dataset.attr[self.train_dataset.attr[:,self.target_attr_idx] != self.train_dataset.attr[:,self.bias_attr_idx]]))+"]")
        print("[Conflicting Samples in Validation Data]["+str(len(self.valid_dataset.attr[self.valid_dataset.attr[:,self.target_attr_idx] != self.valid_dataset.attr[:,self.bias_attr_idx]]))+"]")
        print("[Conflicting Samples in Test Data]["+str(len(self.test_dataset.attr[self.test_dataset.attr[:,self.target_attr_idx] != self.test_dataset.attr[:,self.bias_attr_idx]]))+"]")

        print("[Number of samples in each class]")
        for i in range(self.num_classes):
            print("[Class "+str(i)+"]")
            print("[Training Data]["+str(len(self.train_dataset.attr[self.train_dataset.attr[:,self.target_attr_idx] == i]))+"]")

        self.train_target_attr = self.train_dataset.attr[:, self.target_attr_idx]
        self.train_bias_attr = self.train_dataset.attr[:, self.bias_attr_idx]

        self.train_dataset = IdxDataset(self.train_dataset)
        self.valid_dataset = IdxDataset(self.valid_dataset)    
        self.test_dataset = IdxDataset(self.test_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            **self.loader_config['train'],
            )

        self.test_loader = DataLoader(
            self.test_dataset,
            **self.loader_config['test'],)

        self.valid_loader = DataLoader(
            self.valid_dataset,
            **self.loader_config['valid'],)
         
    def models(self):
        self.model_d = dic_models[self.model_in_d](self.num_classes).to(device)
        self.model_b = dic_models[self.model_in_b](self.num_classes).to(device)
        
        if torch.cuda.device_count() > 1:
            self.model_d = nn.DataParallel(self.model_d)
            self.model_b = nn.DataParallel(self.model_b)

        print("[MODEL D]["+self.model_in_d+"]")
        print("[MODEL B]["+self.model_in_b+"]")
    
    def optimizers(self):
        if 'MNIST' in self.dataset_in:
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(),lr= 0.001, weight_decay=0.0)
            self.optimizer_d = torch.optim.Adam(self.model_d.parameters(),lr= 0.001, weight_decay=0.0)
            self.epochs = 250
        else:
            print('Invalid Dataset')
            exit(0)

        print("[OPTIMIZER]["+str(self.optimizer_d)+"]")
        print("[EPOCHS]["+str(self.epochs)+"]")

        criterion = nn.CrossEntropyLoss(reduction = 'none')
        self.criterion = criterion.to(device)
        bias_criterion = GeneralizedCELoss()
        self.bias_criterion = bias_criterion.to(device)

        self.sample_loss_ema_b = EMA(torch.LongTensor(self.train_target_attr), num_classes=self.num_classes, alpha=0.9)
        self.sample_loss_ema_d = EMA(torch.LongTensor(self.train_target_attr), num_classes=self.num_classes, alpha=0.9)

    '''
    Simple
    '''
    def train_simple(self):

        test_accuracy = -1.0
        test_ = -1.0
        test_accuracy_epoch = -1.0
        valid_accuracy_best = -1.0
        model_path = os.path.join('models',self.name_run)
        pbar = tqdm(range(self.epochs * len(self.train_loader)), desc='Training', ncols=100, leave=False, position=0)
        evaluate_accuracy = dic_functions['Simple']

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                data = data.to(device)
                attr = attr.to(device)
                
                label = attr[:, self.target_attr_idx]

                logit_d = self.model_d(data)
                loss = torch.mean(self.criterion(logit_d, label))
                
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_d.step()
                pbar.update(1)

            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.train_loader, self.target_attr_idx, device)
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.test_loader, self.target_attr_idx, device)
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.valid_loader, self.target_attr_idx, device)

            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)

            if valid_accuracy_best == valid_accuracy_epoch:
                test_accuracy = test_accuracy_epoch

            test_ = max(test_, test_accuracy_epoch)
            os.system('cls' if os.name == 'nt' else 'clear')

            pbar.set_description('Epoch: {}/{} Train Acc: {:.4f} Test Acc Epoch: {:.4f} Best Test Acc: {:.4f}'.format(step, self.epochs, train_accuracy_epoch, test_accuracy_epoch, test_))
            
            if test_accuracy_epoch == test_:
                torch.save(self.model_d, model_path+'_model_d.pt')

        return train_accuracy_epoch, test_accuracy_epoch, test_

    '''
    Ours
    '''
    def train_ours(self):
        mix_up = mixup_data(self.mix_up_val)
        test_accuracy = -1.0
        test_ = -1.0
        test_accuracy_epoch = -1.0
        model_path = os.path.join('models',self.name_run)
        valid_accuracy_best = -1.0
        evaluate_accuracy = dic_functions['ours']

        pbar = tqdm(range(self.epochs * len(self.train_loader)), desc='Training', ncols=100, leave=False, position=0)

        for step in range(1, self.epochs+1):
            for ix, (index,data,attr) in enumerate(self.train_loader):
                flag = True
                data = data.to(device)
                attr = attr.to(device)
                
                label = attr[:, self.target_attr_idx]
                
                logit_b = self.model_b(data)
                logit_d = self.model_d(data)
                loss_b = self.criterion(logit_b, label).cpu().detach()
                loss_d = self.criterion(logit_d, label).cpu().detach()

                self.sample_loss_ema_b.update(loss_b, index)
                self.sample_loss_ema_d.update(loss_d, index)
                
                loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()
                
                label_cpu = label.cpu()

                for c in range(self.num_classes):
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                

                if self.thresh == 'mean':
                    indices = torch.where(loss_weight >= torch.mean(loss_weight))[0]
                elif self.thresh == 'median':
                    indices = torch.where(loss_weight >= torch.median(loss_weight))[0]
                else:
                    self.thresh = float(self.thresh)
                
                    indices = torch.argsort(loss_weight)
                    indices = indices[int(self.thresh*len(indices)):]

                if len(indices) == 0:
                    flag = False
                    print('No conflict samples found')
                else:
                    data_conflict = data[indices]
                    label_conflict = label[indices]

                    for i in range(len(data_conflict)):

                        data_random = data[label == label_conflict[i].item()]

                        rand_value = np.random.randint(0, len(data_random))

                        data_conflict[i] = mix_up(data_conflict[i], data_random[rand_value])
            
                loss_weight = loss_weight.to(device)
               
                loss_b_update = self.bias_criterion(logit_b, label)
                loss_d_update = self.criterion(logit_d, label) * loss_weight
                loss = loss_b_update.mean() + loss_d_update.mean()

                '''
                Reweighting Conflicts
                ''' 
                if flag and step > 0:     
                    
                    try:
                        logit_b_c = self.model_b(data_conflict)
                        logit_d_c = self.model_d(data_conflict)
                        loss_b_c = self.criterion(logit_b_c, label_conflict).cpu().detach()
                        loss_d_c = self.criterion(logit_d_c, label_conflict).cpu().detach()

                        self.sample_loss_ema_b.update(loss_b_c, index[indices])
                        self.sample_loss_ema_d.update(loss_d_c, index[indices])
                        
                        loss_b_c = self.sample_loss_ema_b.parameter[index[indices]].clone().detach()
                        loss_d_c = self.sample_loss_ema_d.parameter[index[indices]].clone().detach()
                        
                        label_cpu = label_conflict.cpu()

                        for c in range(self.num_classes):
                            class_index = np.where(label_cpu == c)[0]
                            max_loss_b = self.sample_loss_ema_b.max_loss(c)
                            max_loss_d = self.sample_loss_ema_d.max_loss(c)
                            loss_b_c[class_index] /= max_loss_b
                            loss_d_c[class_index] /= max_loss_d
                        
                        loss_weight_conflict = loss_b_c / (loss_b_c + loss_d_c + 1e-8)

                        loss_weight_conflict = loss_weight_conflict.to(device)

                        loss_d_update_c = self.criterion(logit_d_c, label_conflict) * loss_weight_conflict
                        loss += self.loss_contr * loss_d_update_c.mean()
                    except:
                        print('Error in Reweighting Conflicts')
                        print(data_conflict.shape)
                        print(label_conflict.shape)
                        pass

                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                self.optimizer_d.step()
                pbar.update(1)

            train_accuracy_epoch = evaluate_accuracy(self.model_d, self.train_loader, self.target_attr_idx, device)
            
            test_accuracy_epoch = evaluate_accuracy(self.model_d, self.test_loader, self.target_attr_idx, device)
            valid_accuracy_epoch = evaluate_accuracy(self.model_d, self.valid_loader, self.target_attr_idx, device)

            valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)

            if valid_accuracy_best == valid_accuracy_epoch:
                test_accuracy = test_accuracy_epoch

            test_ = max(test_, test_accuracy_epoch)
            os.system('cls' if os.name == 'nt' else 'clear')

            pbar.set_description('Epoch: {}/{} Train Acc: {:.4f} Test Acc Epoch: {:.4f} Best Test Acc: {:.4f}'.format(step, self.epochs, train_accuracy_epoch, test_accuracy_epoch, test_))
            
            if test_accuracy_epoch == test_:
                torch.save(self.model_d, model_path+'_model_d.pt')
                torch.save(self.model_b, model_path+'_model_b.pt')


        return train_accuracy_epoch, test_accuracy_epoch, test_

    '''
    Get Results
    '''
    def get_results(self, seed):
        set_seed(seed)
        print('[Training][{}]'.format(self.run_type))
        self.datasets()
        if self.reduce:
            self.reduce_data()
        self.dataloaders()
        self.models()
        self.optimizers()
        self.model_d = self.preprocess_model(self.model_d, self.train_loader, self.epoch_preprocess, self.preprocess)
        
        if self.run_type == 'simple':
            a,b,c = self.train_simple()
            self.store_results(a,b,c)
        elif self.run_type == 'ours':
            a,b,c = self.train_ours()
            self.store_results(a,b,c)
        else:
            print('Invalid run type')
            return