import numpy as np

class GenomeBinary():
    """   二进制编码基因组
        pop_size---种群规模
        chrom_lne---染色体长度
        code_len---二进制编码长度
    """

    def __init__(self,pop_size,chrom_len,code_len=16):
        self.pop_size = pop_size
        self.chrom_size = chrom_len
        self.code_len = code_len
        self.best = None

        self.data = np.random.random((pop_size, chrom_len*code_len)) < 0.5
        self.Binary_template = np.zeros(code_len)
        for i in range(code_len):
            self.Binary_template[i] = (2**i) / 2**code_len

    def select(self, fitness_array):
        """ 选择 """

        indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size,
                                   p=fitness_array/fitness_array.sum())
        self.data[:], fitness_array[:] = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(np.delete(np.arange(self.pop_size), idx), size=1)
                # crossopint = np.random.randint(0, 2, self.pop_size*self.code_len).astype(bool)
                crossopint = (np.random.random(self.chrom_size*self.code_len)) < np.random.rand()
                self.data[idx, crossopint], self.data[idx_other, crossopint] = \
                    self.data[idx_other, crossopint], self.data[idx, crossopint]

    def mutate(self, mutate_prob):
        """ 突变 """
        for idx in range(self.pop_size):
            if np.random.rand() < mutate_prob:
                mutate_position = np.random.choice(np.arange(self.chrom_size*self.code_len), size=1)
                self.data[idx, mutate_position] = ~self.data[idx, mutate_position]

    def to_view(self, chrom):
        """ 将编码转换为01之间的实值 """
        return np.sum(chrom.reshape(self.chrom_size, self.code_len)*self.Binary_template, axis=1)

    def to_real(self, idx, bound):
        """ 编码转换为真实值 """
        chrom = self.to_view(self.data[idx])
        return (bound[1] - bound[0]) * chrom + bound[0]


class Genetic_Algorithm:
    """ 遗传算法实现：
        pop_size---种群规模
        chrom_lne---染色体长度
        bound---染色体真实值上下界
        fitness_func---适应性函数
        GenomeClass---基因组编码形式
        cross_prob---交叉概率
        mutate_prob---突变概率
    """
    def __init__(self, pop_size, chrom_len, bound, fitness_func, GenomeClass, cross_prob, mutate_prob):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.bound = bound
        self.fitness_func = fitness_func
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.Genome = GenomeClass(pop_size=pop_size, chrom_len=chrom_len)

        self.fitness_array = np.zeros(pop_size)
        self.best_pri = np.zeros(chrom_len * self.Genome.code_len)
        self.best_idx_fit = 0
        self.update_fitness()
        self.best_fitness = 0
        self.update_records()

    def update_records(self):
        """ 更新最佳记录  """
        self.best_idx = np.argmax(self.fitness_array)
        self.Genome.best = self.Genome.data[self.best_idx].copy()
        self.best_fitness = self.fitness_array[self.best_idx].copy()

    def replace(self):
        """ 使用前代最佳替换本代最差  """
        worst_idx = np.argmin(self.fitness_array)
        self.Genome.data[worst_idx] = self.Genome.best.copy()
        self.fitness_array[worst_idx] = self.best_fitness

    def update_fitness(self):
        """ 重新计算适应度函数 """
        for idx in range(self.pop_size):
            self.fitness_array[idx] = self.fitness_func(self.Genome.to_real(idx, self.bound))
        best_idx = np.argmax(self.fitness_array)
        worst_idx = np.argmin(self.fitness_array)
        self.Genome.data[worst_idx] = self.best_pri.copy()
        self.fitness_array[worst_idx] = self.best_idx_fit
        self.best_pri = self.Genome.data[best_idx].copy()
        self.best_idx_fit = self.fitness_array[best_idx]
        print(self.best_idx_fit)

    def result(self):
        """ 输出最佳染色体 """
        self.best_params = self.Genome.to_view(self.Genome.best)
        return self.best_params * (self.bound[1] - self.bound[0]) + self.bound[0]

    def genetic(self, num_gen, log=True):
        """ 开始进行遗传算法 """
        for i in range(num_gen):
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<{}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(i+1))
            self.Genome.select(self.fitness_array)
            self.Genome.cross(self.cross_prob)
            self.Genome.mutate(self.mutate_prob)
            self.update_fitness()
            self.replace()

            if self.fitness_array[np.argmax(self.fitness_array)] > self.best_fitness:
                # self.replace()
                self.update_records()

            if log:
                print('Evolution {}, Best: {}, Average: {}'.format(
                   i+1, self.fitness_array.max(), self.fitness_array.mean()))

        print("Best fitness: {}".format(self.fitness_func(self.result())))


if __name__ == '__main__':
    def F(x):
        return x + 10*np.sin(5*x) + 7*np.cos(4*x)

    def calculate_fitness(x):
        return F(x) + 50

    POP_SIZE = 30
    CHROM_LEN = 1
    X_BOUND = (-2, 5)
    bound = np.zeros((2, CHROM_LEN))
    bound[0] = X_BOUND[0] * np.ones(CHROM_LEN)
    bound[1] = X_BOUND[1] * np.ones(CHROM_LEN)

    # Evolution
    ga = Genetic_Algorithm(POP_SIZE, CHROM_LEN, bound, calculate_fitness,
            GenomeClass=GenomeBinary, cross_prob=0.5, mutate_prob=0.1)
    ga.genetic(50, log=True)

    # Plot
    import matplotlib.pyplot as plt
    x_axis = np.linspace(*X_BOUND, 200)
    print("Best fitness: {}, target: {}".format(calculate_fitness(
        ga.result())[0], calculate_fitness(x_axis[np.argmax(F(x_axis))])))
    plt.plot(x_axis, F(x_axis))
    plt.scatter(ga.result(), F(ga.result()), color='r')
    plt.show()

