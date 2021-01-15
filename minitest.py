import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
input_vec = torch.rand(5,5)
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.output = nn.sequential(
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,3)
        )
    def forward(self,input):
        return self.features(input)

raw_model = mynet()
bound_model = BoundedModule(raw_model,input_vec)
num_actions = 3
batchsize = 5
label = torch.tensor([0,2,1,1,0])
bnd_state = BoundedTensor(input_vec, PerturbationLpNorm(norm=np.inf, eps=0.1))

c = torch.eye(3).type_as(input_vec)[label].unsqueeze(1) - torch.eye(3).type_as(input_vec).unsqueeze(0)
I = (~(label.data.unsqueeze(1) == torch.arange(3).type_as(label.data).unsqueeze(0)))
c = (c[I].view(input_vec.size(0), 2, 3))

pred = bound_model(input_vec)
basic_bound,_ = bound_model.compute_bounds(IBP=False,method='backward')
advance_bound,_ = bound_model.compute_bounds(C=c, IBP=False,method='backward')
print(basic_bound.detach().numpy())
print(advance_bound.detach().numpy())