
# coding: utf-8

# In[1]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import json
data = pd.read_csv('News_Final.csv')[0:1000] #导入csv文件
data


# In[2]:


y = list(data['SentimentHeadline'])
y


# In[3]:


#查看字典
with open('Words_Dict.json','r') as f:
    json1=json.load(f)
with open('IDs_Dict.json','r') as f:
    json2=json.load(f)
with open('Words_TF_DF.json','r') as f:
    json3=json.load(f)


# In[4]:


Data=[]
for i in json3:
    a0=[0 for i in range(len(json1))]
    for j in json3[i]:
        a0[int(j)]=json3[i][j]
    Data.append([a0,y[int(i)]])


# In[5]:


Data=np.array(Data)


# In[6]:


Data


# In[7]:


np.random.shuffle(Data)#将数据集打乱，实现随机分割


# In[8]:


Data


# In[9]:


X=[]
Y=[]
for i in Data:
    X.append(i[0])
    Y.append([i[1]])
X=np.array(X)
Y=np.array(Y)


# In[10]:


X


# In[11]:


Y


# In[12]:


#将数据集分为训练集、测试集、验证集、小训练集
training_set=[X[0:800],Y[0:800]]
testing_set=[X[800:1000],Y[800:1000]]
validation_set=[[X[0:160],Y[0:160]],
                [X[160:320],Y[160:320]],
                [X[320:480],Y[320:480]],
                [X[480:640],Y[480:640]],
                [X[640:800],Y[640:800]]]
straining_set=[[X[160:800],Y[160:800]],
               [np.concatenate((X[0:160],X[320:800]),axis=0),np.concatenate((Y[0:160],Y[320:800]),axis=0)],
               [np.concatenate((X[0:320],X[480:800]),axis=0),np.concatenate((Y[0:320],Y[480:800]),axis=0)],
               [np.concatenate((X[0:480],X[640:800]),axis=0),np.concatenate((Y[0:480],Y[640:800]),axis=0)],
               [X[0:640],Y[0:640]]]


# In[13]:


#回归类，用静态方法
class Reg(object):
    def loss(X,Y,w,c,lamda):
        rows=len(Y)
        C=np.array([[c]*rows]).T
        return (1/2*(np.linalg.norm(np.dot(X,w)+C-Y))**2+1/2*lamda*(np.linalg.norm(w))**2)
    @staticmethod
    def RMSE(X,Y,w,c):
        rows=len(Y)
        C=np.array([[c]*rows]).T
        return np.linalg.norm(np.dot(X,w)+C-Y)/(rows**(1/2))
    @staticmethod
    def train(training_set,testing_set,lamda):
        epoch=0
        Dnum=len(training_set[1])
        wnum=training_set[0].shape[1]
        w=np.random.randn(wnum,1)
        c=np.random.rand()
        loss_history=[]
        while epoch<1000:
            delta_Y=np.dot(training_set[0],w)+c*np.ones([Dnum,1])-training_set[1]
            dL_dw=np.dot(training_set[0].T,delta_Y)+lamda*w
            dL_dc=np.dot(np.ones(Dnum),delta_Y)
            alpha=Reg.get_alpha(training_set[0],training_set[1],w,c,lamda,0.1,dL_dw,dL_dc)#初始alpha=0.1
            w=w-alpha*dL_dw
            c=c-alpha*dL_dc
            c=c[0]
            loss_history.append(Reg.loss(X,Y,w,c,lamda))#存储损失函数
            epoch+=1
        return w,c,Reg.RMSE(testing_set[0],testing_set[1],w,c),loss_history
    #回溯线算法调整步长
    def get_alpha(X,Y,w,c,lamda,alpha,dL_dw,dL_dc):
        ce=0.0001
        p=0.5
        now=Reg.loss(X,Y,w,c,lamda)
        nxt=Reg.loss(X,Y,w-alpha*dL_dw,c-alpha*dL_dc,lamda)
        count=30
        while nxt<now:
            alpha/=p
            nxt=Reg.loss(X,Y,w-alpha*dL_dw,c-alpha*dL_dc,lamda)
            count-=1
            if count==0:
                break
        count=50
        #Armijo condition 𝐿(𝑤−𝛼𝛿𝐿)≤𝐿(𝑤)−𝑐𝛼𝛿𝐿^𝑇𝛿𝐿
        while nxt>now-ce*alpha*((np.linalg.norm(dL_dw))**2+dL_dc**2):
            alpha*=p
            nxt=Reg.loss(X,Y,w-alpha*dL_dw,c-alpha*dL_dc,lamda)
            count-=1
            if count==0:
                break
        return alpha


# In[14]:


#交叉验证
RMSEs=[]
for lamda in [0.1*i for i in range(1,11)]:
    errors=[]
    for j in range(5):
        w1,c1,loss1,loss_history1=Reg.train(straining_set[j],validation_set[j],0.1)
        errors.append(loss1)
    print(errors)
    RMSEs.append(errors)


# In[15]:


#对五个训练集、验证集的误差取平均值
RMSEs_mean=list(map(np.mean,RMSEs))
Lambda=RMSEs_mean.index(min(RMSEs_mean))*0.1+0.1#取误差最小对应的正则化系数


# In[16]:


RMSEs_mean


# In[17]:


plt.figure()  
plt.plot([0.1*i for i in range(1,11)],RMSEs_mean)
plt.xlabel('lamda')
plt.ylabel('RMSEs_mean')
# 添加图表标题：title
plt.title('Line Chart for lamda-RMSEs_mean')
plt.show()


# In[18]:


Lambda


# In[19]:


w,c,RMSE,loss_history=Reg.train(training_set,testing_set,Lambda)#用梯度下降法的训练函数


# In[20]:


w


# In[21]:


c


# In[22]:


RMSE


# In[23]:


#误差趋势图
plt.figure()  
plt.plot([epoch for epoch in range(1000)],loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Line Chart for the change of Loss')
plt.show()


# In[24]:


list(w.T[0]).index(max(list(w.T[0])))


# In[25]:


list(w.T[0]).index(min(list(w.T[0])))


# In[26]:


list(json2.values())[156]


# In[27]:


list(json2.values())[661]

