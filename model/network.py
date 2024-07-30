import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNet import ResNet
class SCS(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32,k=1):
        super(SCS, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=k + i*2 , stride=stride, padding=i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feas=[]
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            feas.append(fea)
        feas = torch.cat(feas, dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class PSP(nn.Module):
    def __init__(self, features, out_features, sizes=(1,2,3,6)):
        super(PSP, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h,w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        x=torch.cat(priors, 1)
        x = self.bottleneck(x)
        x= self.relu(x)
        return x




class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.hiden_dim=channel
        self.q_w = nn.Linear(channel,32)
        self.k_w = nn.Linear(channel,32)

    def pcc(self,x):
        mean = torch.mean(x, dim=1, keepdim=True)
        ct = x - mean
        std = torch.std(x, dim=1, keepdim=True)
        cov_matrix = torch.matmul(ct, ct.transpose(1, 2))
        pcc = cov_matrix / (torch.matmul(std, std.transpose(1, 2)))
        return pcc

    def forward(self, x):
        pcc_w=self.pcc(x)
        q=self.q_w(pcc_w)
        k=self.k_w(pcc_w)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))/(self.hiden_dim**0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        x=attention_weights@x
        return x,attention_weights



class TopologySelfAwareAttention(nn.Module):
    def __init__(self):
        super(TopologySelfAwareAttention,self).__init__()
        self.attentions = nn.ModuleList([
            ChannelAttention(17) for _ in range(3)
        ])

        self.c_attention=ChannelAttention(17)
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16, 5, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(5, 16, bias=False),
            nn.Sigmoid()
        )


    def forward(self,x):
        outputs = []
        atts=[]
        for i, attention in enumerate(self.attentions):
            output_i,_at = attention(x[:, i])
            outputs.append(output_i)
            atts.append(_at)
        x=torch.stack(outputs, dim=1)
        return x,atts


class MutilModal(nn.Module):
    def __init__(self,type='classification'):
        super(MutilModal,self).__init__()
        self.eeg_modal=TopologySelfAwareAttention()
        self.expand=nn.Sequential(
            nn.ConstantPad2d((10,10,0,16), 0)
        )
        self.psp= PSP(3,16)
        self.scs = SCS(features=16, WH=36 * 36, M=3, G=16, r=2)
        self.res_net= ResNet(16,32,[2,2,2,2])
        if type=='classification':
            self.classification=nn.Sequential(
                nn.Linear(256,2),
                nn.Softmax(dim=1)
            )
        else:
            self.classification = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
    def split_band_fusion(self,e_data,o_data):
        o_data = o_data.unsqueeze(1).repeat(1, 3, 1, 1)
        e_data = self.expand(e_data)
        x = torch.cat((e_data, o_data), dim=2)
        return x
    def forward(self,e_data,o_data):
        e_data=self.eeg_modal(e_data)

        o_data=o_data.unsqueeze(1).repeat(1,3,1,1)
        e_data = self.expand(e_data)
        x=self.split_band_fusion(e_data,o_data)
        x=self.psp(x)
        x=self.scs(x)
        x=self.res_net(x)
        x=self.classification(x)
        return x