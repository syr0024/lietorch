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

a = torch.randn(1, 3)

print(vec)

# create SO3 object from quaternion (differentiable w.r.t q)
T = SE3.InitFromVec(wq)
R = SO3.InitFromVec(q)
SEK3_1 = SEK3.InitFromVec(vec)
SEK3_2 = SEK3.InitFromCat(R, vec[...,4:])

print(R.matrix()[...,:3,:3].bmm(a.unsqueeze(-1)))
print(R*a)

Adj = SEK3_1.adj_mat()
# k = 3+3*6
# Adj = torch.zeros(1,k,k)
# for i in range(k):
#   Adj[:,:,i] = SEK3_1.adj(torch.eye(k).unsqueeze(0)[:,:,i])

dX = SEK3_1# * SEK3_2.inv()
loss = dX.log().norm(dim=-1).sum()

loss.backward()

a = .2*torch.randn(2,3,SE3.manifold_dim).double()
b = SE3.exp(a)

# i = SEK3.InitFromVec(torch.zeros((1,22)))
# print(i.matrix())
#
# print(SE3.Identity(1).data)
# print(SEK3.Identity(1).data)
# print(SE3.Identity(1).matrix())
# print(SEK3.Identity(1).matrix())
#
# # 4x4 transformation matrix (differentiable w.r.t R)
# T_mat = T.matrix()
# R_mat = R.matrix()
# SE_mat = SEK3_1.matrix()
#
# print(T.data)
# print(SEK3_1.data)
# print(T_mat)
# print(R_mat)
# print(SE_mat)
#
# print(T.log())
# print(R.log())
# print(SEK3_1.log())
#
# a = .2*torch.randn(1,SEK3.manifold_dim).double()
# b = torch.cat((a[...,3:6],a[...,:3]), dim=-1)
# a.requires_grad_()
# b.requires_grad_()
# print(SEK3.exp(a).data)
# print(SE3.exp(b).data)
#
# # map back to quaterion (differentiable w.r.t R)
# T_vec = T.vec()
# R_vec = R.vec()

#
#
# def hat(v):
#   """Convert a vector to a skew-symmetric matrix (hat operator)"""
#   N, dim = v.shape
#   if dim != 3:
#     raise ValueError("Input vectors have to be 3-dimensional.")
#
#   h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)
#
#   x, y, z = v.unbind(1)
#
#   h[:, 0, 1] = -z
#   h[:, 0, 2] = y
#   h[:, 1, 0] = z
#   h[:, 1, 2] = -x
#   h[:, 2, 0] = -y
#   h[:, 2, 1] = x
#
#   return h
#
# # Define the vector v with requires_grad=True to compute gradients
# v = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
#
# # Compute the hat operator
# hat_v = hat(v)
#
# hat_v = hat_v.bmm(R_mat[...,:3,:3])
#
# # Define a simple scalar function of the hat matrix for which we want the gradient
# # For example, we'll take the Frobenius norm of the hat matrix as the scalar function
# scalar_function = torch.mean(hat_v)
#
# # Compute the gradient with respect to v
# scalar_function.backward()
#
# # The gradient is stored in v.grad
# print("Gradient of the scalar function with respect to v:")
# print(v.grad)
