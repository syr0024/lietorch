import torch
from lietorch import SO3, SEK3


# random quaternion
vec = torch.randn(1, 4+3*6)
q = vec[...,:4]
q = q / q.norm(dim=-1, keepdim=True)
vec[...,:4] = q
vec.requires_grad_()
q.requires_grad_()

# create SO3 object from quaternion (differentiable w.r.t q)
T = SEK3.InitFromVec(vec)
R = SO3.InitFromVec(q)

# 4x4 transformation matrix (differentiable w.r.t R)
T_mat = T.matrix()
R_mat = R.matrix()

print(T_mat)
print(R_mat)

print(T.log())
print(R.log())

# map back to quaterion (differentiable w.r.t R)
T_vec = T.vec()
R_vec = R.vec()

