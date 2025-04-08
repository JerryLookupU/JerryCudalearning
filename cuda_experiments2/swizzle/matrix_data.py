import numpy as np
import torch


# a = np.array([[1,2,3,4],[5,6,7,8]])
# b = np.array([[1,2,3,4],[5,6,7,8]])
################
# 矩阵运算
#                   1  5  9   13
#                   2  6  10  14
#                   3  7  11  15
#                   4  8  12  16
#     1  2  3  4         
#     5  6  7  8
#     9 10 11 12    
#     13 14 15 16
e1 = np.array([[1,2,3,4],[5,6,7,8]])
e2 = np.array([[1,2,3,4],[5,6,7,8]])

x1 = np.random.randint(1,10,size=(1024,48))
x2 = np.random.randint(1,10,size=(48,1024))
def matrix_dot(x,y,share_shape=[8,8]):
    z = np.zeros((x.shape[0],y.shape[1]))
    middle_matrix_x = np.zeros((share_shape))
    middle_matrix_y = np.zeros((share_shape))
    # 首先计算x1 能划分成多少个 8*8的矩阵块
    shape_shape_block = share_shape[0]*share_shape[1]
    x_block = x.shape[0]*x.shape[1]//shape_shape_block
    y_block = y.shape[0]*y.shape[1]//shape_shape_block
    # 计算每个矩阵块的起始位置
    # x 的 z轴划分 和y轴划分
    x_x = x.shape[0] // share_shape[0]
    x_y = x.shape[1] // share_shape[1]
    y_x = y.shape[0] // share_shape[0]
    y_y = y.shape[1] // share_shape[1]

    # 分块计算循环
    for i in range(x_x):
        for j in range(y_y):
            # 计算当前分块在原始矩阵中的起始坐标
            x_row_start = i * share_shape[0]
            x_col_start = j * share_shape[1]
            
            y_row_start = j * share_shape[0]
            y_col_start = i * share_shape[1]

            # 加载分块数据到中间矩阵 (模拟shared memory)
            middle_matrix_x = x[x_row_start:x_row_start+share_shape[0], 
                               x_col_start:x_col_start+share_shape[1]]
            
            middle_matrix_y = y[y_row_start:y_row_start+share_shape[0],
                               y_col_start:y_col_start+share_shape[1]]

            # 计算分块乘积并累加
            z_block = np.dot(middle_matrix_x, middle_matrix_y)
            
            # 计算结果写回对应位置
            z[x_row_start:x_row_start+share_shape[0], 
              y_col_start:y_col_start+share_shape[1]] += z_block

    return z


if __name__ == "__main__":
    # 测试分块计算
    test_x = np.random.randint(1,5,size=(16,16))
    test_y = np.random.randint(1,5,size=(16,16))
    
    # 使用不同分块大小测试
    for tile_size in [8, 16]:
        print(f"\nTesting with tile size {tile_size}x{tile_size}")
        
        # 分块计算结果
        block_result = matrix_dot(test_x, test_y, [tile_size, tile_size])
        # 标准矩阵乘法结果
        direct_result = test_x @ test_y
        
        # 验证结果一致性
        if np.allclose(block_result, direct_result, atol=1e-5):
            print(f"Success! {tile_size}x{tile_size} block calculation matches direct multiplication")
        else:
            print("Validation failed!")
            print("Block result:\n", block_result)
            print("Direct result:\n", direct_result)
    
