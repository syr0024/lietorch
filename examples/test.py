import torch
from lietorch import SO3, SEK3, SE3


# random quaternion
vec = torch.randn(1, 4+3*6)
q = vec[...,:4]
q = q / q.norm(dim=-1, keepdim=True)
vec[...,:4] = q
wq = torch.cat((vec[...,4:7], q), dim=-1)
vec.requires_grad_()
q.requires_grad_()
wq.requires_grad_()

print(vec)

# create SO3 object from quaternion (differentiable w.r.t q)
T = SEK3.InitFromVec(vec)
R = SO3.InitFromVec(q)
SE = SE3.InitFromVec(wq)

# 4x4 transformation matrix (differentiable w.r.t R)
T_mat = T.matrix()
R_mat = R.matrix()
SE_mat = SE.matrix()

print(T_mat)
print(R_mat)
print(SE_mat)

print(T.log())
print(R.log())
print(SE.log())

a = .2*torch.randn(1,SEK3.manifold_dim).double()
b = torch.cat((a[...,3:6],a[...,:3]), dim=-1)
a.requires_grad_()
b.requires_grad_()
print(SEK3.exp(a).data)
print(SE3.exp(b).data)

# map back to quaterion (differentiable w.r.t R)
T_vec = T.vec()
R_vec = R.vec()

