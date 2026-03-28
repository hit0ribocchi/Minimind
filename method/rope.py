import torch 

# x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([10,20,30,40,60])

# condition=x>3
# #不符合条件的用y里的元素来替换x的元素并输出
# result=torch.where(condition,x,y)     # Output： tensor([10, 20, 30,  4,  5])
# print(result)

#---------------------------------------------------#

# #生成等差数列，0-10的步长为2的张量
# t = torch.arange(0,10,2)    # Output： tensor([0, 2, 4, 6, 8])
# print(t)

#---------------------------------------------------#

# t1=torch.tensor([1,2,3])
# t2=torch.tensor([4,5,6])
# #外积
# t=torch.outer(t1,t2)
# # tensor([[ 4,  5,  6], [ 8, 10, 12], [12, 15, 18]])
# print(t)

#---------------------------------------------------#
