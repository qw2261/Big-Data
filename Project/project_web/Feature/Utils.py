import torch



def make_one_hot(labels, class_num=7):
    y = torch.eye(class_num) 
    return y[labels].type(torch.LongTensor)
