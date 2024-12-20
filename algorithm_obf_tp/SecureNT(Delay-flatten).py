import numpy as np

def l2_loss(a, b):
    """L2损失函数，用来衡量分布a和b的差距"""
    return np.sum((a - b) ** 2)

def gradient_step(a, b, learning_rate=0.01):
    """
    使用梯度下降法更新分布a，目标是逼近分布b。
    假设我们最小化L2损失函数，即 (a - b) ^ 2。
    
    :param a: 源分布
    :param b: 目标分布
    :param learning_rate: 学习率
    :return: 更新后的分布a
    """
    grad = 2 * (a - b)  # L2损失的梯度
    a_new = a - learning_rate * grad  # 进行梯度更新
    
    return a_new

def projection(a, original_sum):
    """
    投影操作，确保更新后的分布a的总和与原始的总和一致。
    :param a: 更新后的分布
    :param original_sum: 原始分布的总和
    :return: 投影后的分布a
    """
    # 计算新的总和
    current_sum = np.sum(a)
    # 调整a，使得其总和与原始总和一致
    a_new = a * (original_sum / current_sum)
    
    return a_new

def gradient_descent_with_projection(a, b, max_iter=1000, learning_rate=0.01, tolerance=1e-9):
    """
    使用梯度下降与投影法进行分布逼近。
    :param a: 源分布
    :param b: 目标分布
    :param max_iter: 最大迭代次数
    :param learning_rate: 学习率
    :param tolerance: 收敛容忍度
    :return: 逼近后的分布a
    """
    original_sum = np.sum(a)  # 记录原始分布a的总和
    initial_loss = l2_loss(a, b)  # 记录初始的损失值
    print(f"Initial Loss: {initial_loss}")
    
    for iter_num in range(max_iter):
        # 进行梯度下降更新
        a_new = gradient_step(a, b, learning_rate)
        
        # 投影到总和不变的空间
        a_new = projection(a_new, original_sum)
        
        # 计算当前的损失值
        current_loss = l2_loss(a_new, b)
        
        # 打印迭代过程中的损失变化（可选）
        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}, Loss: {current_loss}")
        
        # 停止条件：如果损失值降到初始损失的一半以下
        if current_loss <= 0.4 * initial_loss:
            print(f"Converged in {iter_num + 1} iterations")
            break
        
        # 更新a
        a = a_new
    
    return a

