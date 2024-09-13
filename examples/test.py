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
T = SE3.InitFromVec(wq)
R = SO3.InitFromVec(q)
SEK3_1 = SEK3.InitFromVec(vec)
SEK3_2 = SEK3.InitFromCat(R, vec[...,4:])

dX = SEK3_1 * SEK3_2.inv()
loss = dX.log().norm(dim=-1).sum()

loss.backward()

a = .2*torch.randn(2,3,SE3.manifold_dim).double()
b = SE3.exp(a)


# 4x4 transformation matrix (differentiable w.r.t R)
T_mat = T.matrix()
R_mat = R.matrix()
SE_mat = SEK3_1.matrix()

print(T_mat)
print(R_mat)
print(SE_mat)

print(T.log())
print(R.log())
print(SEK3_1.log())

a = .2*torch.randn(1,SEK3.manifold_dim).double()
b = torch.cat((a[...,3:6],a[...,:3]), dim=-1)
a.requires_grad_()
b.requires_grad_()
print(SEK3.exp(a).data)
print(SE3.exp(b).data)

# map back to quaterion (differentiable w.r.t R)
T_vec = T.vec()
R_vec = R.vec()

