# 0. PX-Mixer结果复现
CUDA_VISIBLE_DEVICES=1 python /home/jack/Project/MutiModal/SentimentAnalysis/JackNet/MultimodalSA/train.py

####################################################################################################                                                                  
Epoch 23 - Training loss: 0.0908                                                                                                                                      
mult_acc7:  0.5259219668626403                                                                                                                                        
Accuracy (non-neg/neg)  0.7851416354890433                                                                                                                            
val loss: 0.5579, val acc: 0.7851                                                                                                                                     
test dataset:                                                                                                                                                         
mult_acc7:  0.5199828104856038                                                                                                                                        
Accuracy (non-neg/neg)  0.7911474000859475                                                                                                                            
test loss: 0.5869, test acc: 0.7911                                                                                                                                   
Current patience: 1, current trial: 1.  
####################################################################################################                                                                  
Epoch 24 - Training loss: 0.0814                                                                                                                                      
mult_acc7:  0.5253874933190807                                                                                                                                        
Accuracy (non-neg/neg)  0.7851416354890433                                                                                                                            
val loss: 0.5538, val acc: 0.7851                                                                                                                                     
test dataset:                                                                                                                                                         
mult_acc7:  0.5197679415556511                                                                                                                                        
Accuracy (non-neg/neg)  0.7898581865062312                                                                                                                            
test loss: 0.5901, test acc: 0.7899                                                                                                                                   
Current patience: 0, current trial: 1.                                                                                                                                
Running out of patience, loading previous best model.                                                                                                                 
Current learning rate: 2.5e-05                                                                                                                                        
####################################################################################################                                                                  
Running out of patience, early stopping.
mult_acc7:  0.5279329608938548                                                                                                                                        
mae:  0.54106534                                                                                                                                                      
corr:  0.7614680788582571                                                                                                                                             
mult_acc5:  0.5436183927804039                                                                                                                                        
Classification Report (pos/neg) :                                                                                                                                     
              precision    recall  f1-score   support                                                                                                                 
                                                                                                                                                                      
       False    0.82844   0.77374   0.80015      1348                                                                                                                 
        True    0.87131   0.90530   0.88798      2281                                                                                                                 
                                                                                                                                                                      
    accuracy                        0.85643      3629                                                                                                                 
   macro avg    0.84987   0.83952   0.84407      3629                                                                                                                 
weighted avg    0.85538   0.85643   0.85536      3629                                                                                                                 

Accuracy (pos/neg)  0.8564342794158171                                             
F1: 0.8575112511047523                                   

Classification Report (non-neg/neg) :                                                                                                                       
              precision    recall  f1-score   support                                                                                                       
                                                                                                                                                            
       False    0.67596   0.77374   0.72155      1348                                                                                                       
        True    0.90196   0.84876   0.87455      3306                                                                                                       
                                                                                                                                                                                    
    accuracy                        0.82703      4654                                                                                                       
   macro avg    0.78896   0.81125   0.79805      4654                                                                                                       
weighted avg    0.83650   0.82703   0.83024      4654                                                                                                       

Accuracy (non-neg/neg)  0.8270305113880533                                    
F1: 0.8238251552178933                                                        

# 1. SwimT 用来提取三种模态特征

####################################################################################################
Epoch 24 - Training loss: 0.0454
mult_acc7:  0.49545697487974344
Accuracy (non-neg/neg)  0.7862105825761625
val loss: 0.588, val acc: 0.7862
test dataset:
mult_acc7:  0.48667812634293084
Accuracy (non-neg/neg)  0.7866351525569403
test loss: 0.6428, test acc: 0.7866
Current patience: 0, current trial: 1.
Running out of patience, loading previous best model.
Current learning rate: 1.25e-05
####################################################################################################
Running out of patience, early stopping.
mult_acc7:  0.5165449076063601
mae:  0.5608818
corr:  0.7467203073347745
mult_acc5:  0.5320154705629566
Classification Report (pos/neg) :
              precision    recall  f1-score   support

       False    0.83109   0.73368   0.77935      1348
        True    0.85281   0.91188   0.88136      2281

    accuracy                        0.84569      3629
   macro avg    0.84195   0.82278   0.83035      3629
weighted avg    0.84474   0.84569   0.84347      3629

Accuracy (pos/neg)  0.8456875172223753
F1: 0.847908009565559
Classification Report (non-neg/neg) :
              precision    recall  f1-score   support

       False    0.67325   0.73368   0.70217      1348
        True    0.88728   0.85481   0.87074      3306

    accuracy                        0.81972      4654
   macro avg    0.78027   0.79424   0.78645      4654
weighted avg    0.82529   0.81972   0.82192      4654

Accuracy (non-neg/neg)  0.8197249677696605
F1: 0.8175335173351049






# UR_FUNNY
https://blog.csdn.net/yangyanbao8389/article/details/121703713
碰到一个 变量并不显示具体的网络输出值,而是数据的地址信息 BUG 
解决方案就是bert那一块要使用一下，放入gpu

罪魁祸首

 <!-- elif self.train_config.data == "ur_funny": # Jack Change 0523 
     if "bert" in name:
         param.requires_grad = False -->

其实是bert sent数据值中有负数 导致 bert_sent过去是一个地址值，而非数据值

还有就是dev中存了之前的模型参数，导师bert预训练模型无法正常使用

acc PXMixer 96%

my 91%

不知道哪里出问题了 暂缓！！！

0524 下午5点  人麻了


# 实验总结
(1) AOTransformer MOSEI 可以全面SOTA
(2) AOTransformer MOSI A2 SOTA A7 表现不佳是因为数据量问题
(3) AOTransformer + Dataenhancement 可以显著提高acc

(1) effective of CLIP MAP
right  (share encoder 之前是否会进行feature map的设计)

(2) effective of modality joint
right (specific encoder 是否使用 Transformer (即对齐之前是否有一个特征提取再融合的操作))

(3) effective of modality invarint enhance
right (使用transfomer作为share encoder vs PXMixer使用Linear作为share encoder)


(4) structure of transformer three modal sharing
right (使用MITRL中的二者share还是三者share)











