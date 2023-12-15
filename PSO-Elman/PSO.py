import numpy as np

class PSO:
    def __init__(self,calculate_fitness,size,w,c1,c2,max_v,dim,max_iter):
        """
        calculate_fitness:适应性函数
        size:粒子群的个数
        w:惯性系数
        c1,c2:权重系数
        max_v:最大速度
        dim:粒子所运动的维度空间(即自变量的个数)
        max_iter:最大迭代次数
        """
        self.size=size
        self.w=w
        self.c1=c1
        self.c2=c2
        self.max_v=max_v
        self.dim=dim
        self.max_iter=max_iter
        self.positions=np.random.uniform(-1,1,size=(size,dim)) 
        self.v=np.random.uniform(-0.5,0.5,size=(size,dim))
        self.func=calculate_fitness
        
    def v_update(self,p_best,g_best):
        r1=np.random.random((self.size,1))
        r2=np.random.random((self.size,1))
        self.v=self.w*self.v+self.c1*r1*(p_best-self.positions)+self.c2*r2*(g_best-self.positions)
        # 限制速度
        self.v=np.clip(self.v,-self.max_v,self.max_v)
        # return self.v
    
    def p_update(self,):
        self.positions += self.v
        self.positions = np.clip(self.positions,-5,5)
        # return self.positions
    
    def roll(self,):
        """开始迭代"""
        best_fitness=float(9e10)
        fitness_val_list_group=[]
        p_fitness=self.func(self.positions)    # 初始化个体适应度
        g_fitness=p_fitness.min()   # 初始化集体适应度
        fitness_val_list_group.append(g_fitness)
        
        p_best=self.positions    # 初始化个体的初始位置
        
        g_best=self.positions[p_fitness.argmin()]    # 初始化种群最优位置
        
        for i in range(self.max_iter):
            self.v_update(p_best,g_best)
            self.p_update()
            p_fitness2=self.func(self.positions)
            g_fitness2=p_fitness2.min()
            
#             # 更新每个粒子的历史最优位置和历史最优适应度
#             for j in range(self.size):
#                 if p_fitness2[j]<p_fitness[j]:
#                     p_best[j]=self.positions[j]
#                     p_fitness[j]=p_fitness2[j]

            # # 更新群体的全局历史最优位置和全局历史最优适应度
            if g_fitness2<g_fitness:
                g_fitness=g_fitness2
                g_best=self.positions[p_fitness2.argmin()]
            
            fitness_val_list_group.append(g_fitness)
            
#             print(g_best)
        return g_best
        