import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x):
    return x / torch.pow(torch.sum(torch.pow(x.view(x.shape[0], -1), 2), dim = 1), .5).view(-1, 1, 1, 1) 

def KLDivergence( p, q ):
    return torch.mean(torch.sum(torch.mul(p , (torch.log(p) - torch.log(q))), dim = 1))


class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        
        #model.eval()	
        #x_ptb = x.copy()
        r = torch.randn(x.shape).to(x.device)
        r = l2_normalize(r)
        
        y_out = model(x)
        y_pred = F.softmax(y_out, dim = 1)	
        
        for v_iter in range(self.vat_iter):
        
            r.requires_grad_()
        	
            y_adv_out = model(x + self.xi * r)
            y_adv_pred = F.softmax(y_adv_out, dim = 1)

            adv_dis = KLDivergence(y_adv_pred, y_pred) 
            adv_dis.backward(retain_graph=True)
            r = l2_normalize(r.grad)
            model.zero_grad() 
		
        r_adv = r * self.eps
        y_adv_out = model(x + r_adv)
        y_adv_pred = F.softmax(y_adv_out, dim = 1)
        loss = KLDivergence(y_adv_pred, y_pred )
        
        return loss
        

        


