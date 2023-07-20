import torch
import torch.nn as nn
import torch.nn.functional as F

class FCB(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c * 2,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c * 2)
        self.linear2 = nn.Linear(in_features=in_c * 2,
                                 out_features=in_c,
                                 bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.linear1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = torch.tanh(output)
        output = 1 + output

        return output

class DualAttention(nn.Module):

    def __init__(self, name_DA):

        super().__init__()
        self.name = name_DA

        if self.name == 'resnet12':
            self.in_c = 640
            self.in_m = 25 # w/o flatten
        else:
            self.in_c = 64
            self.in_m = 25

        self.prt_C = FCB(self.in_c)
        self.prt_S = FCB(self.in_m)
        self.qry_C = FCB(self.in_c)
        self.qry_S = FCB(self.in_m)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def add_noise(self, input):
        if self.training:
            noise = ((torch.rand(input.shape).to(input.device) - .5) * 2) * 0.2
            input = input + noise
            input = input.clamp(min=0., max=2.)

        return input

    def dist(self, input, spt=False, normalize=True):

        if spt:
            
            way, c, m = input.shape
            input_C_gap = input.mean(dim=-2).unsqueeze(dim=-2)
            input_S_gap = input.mean(dim=-1).unsqueeze(dim=-1)

            dist_C = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            dist_S = torch.sum(torch.pow(input - input_S_gap, 2), dim=-2)
            
            if normalize:
                dist_C = dist_C / m
                dist_S = dist_S / c
                
            return dist_C, dist_S

        
        else:
            batch, c, m = input.shape
            input_C_gap = input.mean(dim=-2).unsqueeze(dim=-2)
            input_S_gap = input.mean(dim=-1).unsqueeze(dim=-1)

            dist_C = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            dist_S = torch.sum(torch.pow(input - input_S_gap, 2), dim=-2)
            
            if normalize:
                dist_C = dist_C / m
                dist_S = dist_S / c
                
            return dist_C, dist_S

    def weight(self, spt, qry):
        
        
        prt = spt.mean(dim=1)
        # w,c,m

        dist_prt_C, dist_prt_S = self.dist(prt, spt=True)
        dist_qry_C, dist_qry_S = self.dist(qry)
        

        weight_prt_C = self.prt_C(dist_prt_C) # w,c
        weight_prt_S = self.prt_S(dist_prt_S) # w,m
        weight_qry_C = self.qry_C(dist_qry_C) # n,c
        weight_qry_S = self.qry_S(dist_qry_S) # n,m
    


        weight_C = torch.cat((weight_prt_C, weight_qry_C), dim=0) # w+n,c
        weight_S = torch.cat((weight_prt_S, weight_qry_S), dim=0) # w+n,m
        

        return weight_C, weight_S

    def forward(self, spt, qry):
        weight_C, weight_S = self.weight(spt, qry)
        weight_C = self.add_noise(weight_C)
        weight_S = self.add_noise(weight_S)

        return weight_C, weight_S
    
    

    
