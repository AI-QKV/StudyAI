import math
import random

# ==============================================================================
# 阶段一：数据准备与单词级分词
# 之前我们处理的是单个字符，现在我们的基本单位是“单词”。
# ==============================================================================

print("--- 阶段一：准备单词级数据集 ---")

# 步骤1.1：创建一个比"thinking machines"更丰富的“迷你语料库”
# 这个语料库包含了一些重复的结构和模式，为模型学习提供了可能性。
corpus = (
    "the dog saw the cat . the cat saw the mouse . "
    "the mouse ran away . the dog barked . the cat chased the mouse . "
    "the brown dog and the white cat are friends . "
    "the dog likes to bark . the cat likes to chase . "
    "the mouse likes to run . "
    "what did the dog see ? the dog saw the cat ."
)

# 步骤1.2：进行单词级分词 (Tokenization)
# 这是最基础的分词方式：按空格切分。
tokens = corpus.split()

# 步骤1.3：创建单词级别的词汇表
# 使用集合(set)来获取所有不重复的单词，然后排序以保证每次运行的映射一致。
chars = sorted(list(set(tokens)))
vocab_size = len(chars)

# 步骤1.4：创建单词到索引(stoi)和索引到单词(itos)的映射字典
stoi = {word: i for i, word in enumerate(chars)}
itos = {i: word for i, word in enumerate(chars)}

print(f"我们的词汇表 (部分): {chars[:10]}...")
print(f"词汇表大小 (单词数): {vocab_size}")
print(f"单词到索引的映射 'the' -> {stoi['the']}, 'dog' -> {stoi['dog']}")

# 步骤1.5：创建训练用的“输入-标签”数据对
# 这里的逻辑和字符级完全一样，只是单位从字符变成了单词。
block_size = 3  # 上下文长度：模型根据前3个单词来预测第4个单词
dataset = []
for i in range(len(tokens) - block_size):
    context = tokens[i : i + block_size]
    target = tokens[i + block_size]

    # 将单词上下文和目标转换为数字索引
    context_indices = [stoi[word] for word in context]
    target_index = stoi[target]

    dataset.append((context_indices, target_index))

print(f"\n以 block_size={block_size} 为例，生成的数据集样本：")
for i in range(3):
    context_words = ' '.join([itos[j] for j in dataset[i][0]])
    target_word = itos[dataset[i][1]]
    print(f"  输入: {dataset[i][0]} (即 '{context_words}') -> 标签: {dataset[i][1]} (即 '{target_word}')")


# ==============================================================================
# 阶段二：构建模型的“纯Python”零件
# 我们将重用并完善之前纯Python版本的各个模块。
# ==============================================================================

print("\n--- 阶段二：构建模型的纯Python零件 ---")

# 步骤2.1：定义模型超参数
d_model = 16  # 词向量和模型的维度（为了演示，设得比较小）
num_heads = 4  # 多头注意力的头数
num_blocks = 2 # Transformer解码器块的数量
d_ff = d_model * 4 # 前馈网络中间层的维度，通常是d_model的4倍

# 步骤2.2：纯Python数学工具箱 (和之前一样，但注释更清晰)
def random_matrix(rows, cols):
    """创建一个用小的随机数填充的矩阵（列表的列表）。"""
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

def mat_mul(A, B):
    """计算两个2D矩阵A和B的乘积。"""
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    if cols_a != rows_b:
        raise ValueError("矩阵维度不匹配，无法相乘")
    # 初始化一个全零的结果矩阵C
    C = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    # 核心的三层循环矩阵乘法
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    """计算矩阵的转置。"""
    return [list(row) for row in zip(*A)]

def add_vectors(A, B):
    """将两个同样形状的矩阵逐元素相加。这是实现残差连接的核心。"""
    return [[a + b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

def relu(matrix):
    """对矩阵中的每个元素应用ReLU激活函数。"""
    return [[max(0, val) for val in row] for row in matrix]


# 步骤2.3：零件一：词嵌入 (Embedding)
# 功能：将单词的数字索引，转换为一个高维的、有意义的向量。
class Embedding:
    def __init__(self, vocab_size, d_model):
        # 注释：这个权重矩阵就是我们的“查询表”。每一行都代表一个单词的向量。
        # 在真实训练中，这个矩阵里的值是模型需要学习的最重要的参数之一。
        self.weights = random_matrix(vocab_size, d_model)
    
    def forward(self, indices):
        # 注释：根据输入的索引列表，从权重矩阵中“查出”对应的行（词向量）。
        return [self.weights[idx] for idx in indices]

# 步骤2.4：零件二：位置编码 (Positional Encoding)
# 功能：为模型提供单词在句子中的位置信息。
class PositionalEncoding:
    def __init__(self, max_len, d_model):
        pe = [[0.0 for _ in range(d_model)] for _ in range(max_len)]
        div_term = [math.pow(10000.0, (2 * i) / d_model) for i in range(0, d_model, 2)]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / div_term[i//2])
                if i + 1 < d_model:
                    pe[pos][i+1] = math.cos(pos / div_term[i//2])
        self.pe = pe

    def forward(self, seq_len):
        # 注释：返回一个预先计算好的、长度为seq_len的位置编码矩阵。
        return self.pe[:seq_len]

# 步骤2.5：零件三：层归一化 (LayerNorm)
# 功能：稳定训练过程，我们采用Pre-Norm结构，即在进入主模块前进行归一化。
class LayerNorm:
    def __init__(self, d_model, epsilon=1e-5):
        self.epsilon = epsilon
        self.gamma = [1.0] * d_model # 缩放参数
        self.beta = [0.0] * d_model  # 平移参数

    def forward(self, matrix_x):
        normalized_matrix = []
        # 注释：对一个句子中的每个词向量独立进行归一化
        for vec in matrix_x:
            mean = sum(vec) / len(vec)
            variance = sum([(v - mean) ** 2 for v in vec]) / len(vec)
            std = math.sqrt(variance + self.epsilon)
            normalized_vec = [(v - mean) / std for v in vec]
            # 应用gamma和beta，让网络可以学习恢复原始分布
            scaled_vec = [g * nv + b for g, nv, b in zip(self.gamma, normalized_vec, self.beta)]
            normalized_matrix.append(scaled_vec)
        return normalized_matrix

# 步骤2.6：零件四：前馈网络 (FeedForward)
# 功能：进行非线性信息处理和提炼。
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = random_matrix(d_model, d_ff)
        self.b1 = [0.0] * d_ff
        self.W2 = random_matrix(d_ff, d_model)
        self.b2 = [0.0] * d_model

    def forward(self, matrix_x):
        # 第一次线性变换 + 偏置
        linear1_out = [add_vectors([row], [self.b1])[0] for row in mat_mul(matrix_x, self.W1)]
        # ReLU激活
        relu_out = relu(linear1_out)
        # 第二次线性变换 + 偏置
        linear2_out = [add_vectors([row], [self.b2])[0] for row in mat_mul(relu_out, self.W2)]
        return linear2_out

# 步骤2.7：零件五：带掩码的多头注意力 (MaskedMultiHeadAttention) - 核心模块
# 功能：让模型在生成当前词时，关注之前已经生成的词，并从多个角度进行。
class MaskedMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 注释：初始化Q,K,V和最终输出的权重矩阵
        self.Wq = random_matrix(d_model, d_model)
        self.Wk = random_matrix(d_model, d_model)
        self.Wv = random_matrix(d_model, d_model)
        self.Wo = random_matrix(d_model, d_model)
    
    # 内部辅助函数，用于将数据拆分成多个头
    def _split(self, proj, seq_len):
        """将 [seq_len, d_model] 的矩阵拆分成 [num_heads, seq_len, d_k]"""
        return [[proj[i][h*self.d_k : (h+1)*self.d_k] for i in range(seq_len)] for h in range(self.num_heads)]
    
    # 内部辅助函数，用于合并多个头的输出
    def _combine(self, outputs, seq_len):
        """将 [num_heads, seq_len, d_k] 的输出合并成 [seq_len, d_model]"""
        combined = []
        for i in range(seq_len):
            row = []
            for h in range(self.num_heads):
                row.extend(outputs[h][i])
            combined.append(row)
        return combined

    # 内部辅助函数，用于计算Softmax
    def _softmax(self, matrix):
        """对2D列表的每一行计算softmax"""
        result = []
        for row in matrix:
            max_val = max(row) if row else 0.0 # 数值稳定
            exps = [math.exp(v - max_val) for v in row]
            sum_of_exps = sum(exps)
            result.append([e / sum_of_exps if sum_of_exps > 0 else 0.0 for e in exps])
        return result

    def forward(self, x, mask):
        seq_len = len(x)
        # 1. 线性投影，生成Q, K, V
        Q_proj = mat_mul(x, self.Wq)
        K_proj = mat_mul(x, self.Wk)
        V_proj = mat_mul(x, self.Wv)

        # 2. 拆分成多个头
        Q_split = self._split(Q_proj, seq_len)
        K_split = self._split(K_proj, seq_len)
        V_split = self._split(V_proj, seq_len)

        # 3. 对每个头独立计算注意力
        all_heads_outputs = []
        for h in range(self.num_heads):
            Q_h, K_h, V_h = Q_split[h], K_split[h], V_split[h]
            # 3a. 计算分数
            scores = mat_mul(Q_h, transpose(K_h))
            # 3b. 缩放和掩码
            scaled_scores = [[0.0]*seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    scaled_scores[i][j] = scores[i][j] / math.sqrt(self.d_k)
                    if mask[i][j] == 0:
                        scaled_scores[i][j] = -1e9 # 用一个极小的数填充被mask的位置
            # 3c. Softmax
            attention_weights = self._softmax(scaled_scores)
            # 3d. 加权求和
            output_h = mat_mul(attention_weights, V_h)
            all_heads_outputs.append(output_h)
        
        # 4. 合并多头并进行最终线性变换
        concatenated_output = self._combine(all_heads_outputs, seq_len)
        final_output = mat_mul(concatenated_output, self.Wo)
        return final_output


# ==============================================================================
# 阶段三：组装完整的Transformer架构
# ==============================================================================

print("\n--- 阶段三：组装单词级Transformer模型 ---")

# 步骤3.1：定义一个解码器块 (Decoder Block)
# 这是模型的基本重复单元，包含了注意力和前馈网络。
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MaskedMultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask):
        # Pre-Norm结构: 先归一化，再进模块，更稳定
        # 第一个子层：多头注意力 + 残差连接
        # add_vectors(x, ...) 就是残差连接，把输入x直接加回来
        attention_out = self.mha.forward(self.ln1.forward(x), mask)
        x = add_vectors(x, attention_out)

        # 第二个子层：前馈网络 + 残差连接
        ff_out = self.ff.forward(self.ln2.forward(x))
        x = add_vectors(x, ff_out)
        return x
    
# 步骤3.2：定义最终的Transformer语言模型
class TransformerLanguageModel:
    def __init__(self, vocab_size, d_model, num_heads, num_blocks, max_len):
        self.token_embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_len, d_model)

        # 注释：创建N个解码器块的堆叠
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_blocks)]
        
        # 注释：最终的输出头，将模型输出的向量映射回词汇表大小，得到logits
        self.output_head = random_matrix(d_model, vocab_size)

    def forward(self, indices):
        seq_len = len(indices)
        
        # 1. 获取词嵌入和位置编码
        token_embeds = self.token_embedding.forward(indices)
        pos_embeds = self.positional_encoding.forward(seq_len)
        # 2. 将两者相加，得到模型的初始输入
        x = add_vectors(token_embeds, pos_embeds)

        # 3. 创建因果掩码 (Causal Mask)，防止模型看到未来的单词
        mask = [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]

        # 4. 依次通过每一个解码器块
        for block in self.decoder_blocks:
            x = block.forward(x, mask)
        
        # 5. 只取最后一个时间步的输出向量，用于预测下一个词
        last_token_vector = [x[-1]] # 保持为2D列表 [[...]]

        # 6. 通过输出头得到logits
        logits = mat_mul(last_token_vector, self.output_head)[0] # 取出结果
        return logits

# 步骤3.3：定义损失函数
def cross_entropy_loss(logits, target_index):
    """计算交叉熵损失"""
    # 数值稳定版本的softmax
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    probs = [e / sum_exps for e in exps]
    # 计算损失
    loss = -math.log(probs[target_index] + 1e-9) # 加一个极小的数防止log(0)
    return loss


# ==============================================================================
# 阶段四：伪训练与文本生成
# 再次强调：这里没有反向传播，所以模型不会学习，权重始终是随机的。
# ==============================================================================
print("\n--- 阶段四：伪训练与文本生成 ---")

# 步骤4.1：实例化我们的单词级模型
model = TransformerLanguageModel(
    vocab_size=vocab_size, 
    d_model=d_model, 
    num_heads=num_heads, 
    num_blocks=num_blocks, 
    max_len=block_size
)

# 步骤4.2：进行一个伪训练循环，观察损失值
print("\n开始伪训练循环（模型权重不会更新）...")
training_steps = 10
for i in range(training_steps):
    # 从数据集中随机选择一个样本
    sample_context, sample_target = random.choice(dataset)
    
    # 前向传播，得到预测的logits
    predicted_logits = model.forward(sample_context)

    # 计算损失
    loss = cross_entropy_loss(predicted_logits, sample_target)
    
    # 打印日志
    if i % 2 == 0:
        context_words = ' '.join([itos[j] for j in sample_context])
        target_word = itos[sample_target]
        print(f"步骤 {i}: 输入 '{context_words}', 目标 '{target_word}', 计算出的损失 = {loss:.4f}")
print("\n伪训练结束。损失值是随机的，因为它只取决于随机初始化的权重和随机选择的样本。")


# 步骤4.3：文本生成 (Inference)
def generate(model, start_text, max_new_tokens):
    print(f"\n--- 开始文本生成，初始文本: '{start_text}' ---")
    words = start_text.split()
    indices = [stoi[w] for w in words]
    
    for _ in range(max_new_tokens):
        # 保证输入的上下文长度不超过block_size
        context_indices = indices[-block_size:]
        
        # 前向传播，得到下一个词的logits
        logits = model.forward(context_indices)
        
        # Softmax转成概率
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]
        
        # 选择概率最高的单词作为预测结果 (Greedy Sampling)
        next_index = probs.index(max(probs))
        
        # 把预测结果加到序列中，继续下一次预测
        indices.append(next_index)
        
    return ' '.join([itos[i] for i in indices])

# 因为模型未经训练，所以生成结果依然是完全随机和无意义的
generated_text = generate(model, start_text='the dog', max_new_tokens=10)
print(f"\n未经训练的单词级模型生成结果: '{generated_text}'")
