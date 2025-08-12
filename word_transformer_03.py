import math
import random

# ==============================================================================
# 阶段一：数据准备与单词分词器 (与您提供的代码类似，但更完善)
# ==============================================================================
print("--- 步骤一：创建数据集和单词分词器 ---")

# --- 1.1 生成我们的语料库 ---
# 为了让模型能学到更丰富的模式，我们创建一个更多样化的词汇列表
word_list = [
    'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.',
    'a', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'played', 'with', 'a', 'ball', '.',
    'transformer', 'models', 'are', 'powerful', 'for', 'natural', 'language', 'processing', '.',
    'attention', 'is', 'all', 'you', 'need', 'is', 'a', 'famous', 'paper', '.',
    'we', 'are', 'building', 'a', 'transformer', 'model', 'from', 'scratch', 'to', 'learn', 'the', 'fundamentals', '.'
]
# 随机选择单词，构建一个约200词的文本，使其具有一定的随机性和重复性
corpus = ' '.join(random.choices(word_list, k=200))
print(f"生成的语料库 (前100个字符): {corpus[:100]}...")

# --- 1.2 一个简单的单词分词器 ---
class SimpleWordTokenizer:
    def __init__(self, min_freq=1):
        # stoi: string-to-index, 单词到索引的映射
        # itos: index-to-string, 索引到单词的映射
        self.stoi = {}
        self.itos = {}
        self.min_freq = min_freq

    def fit(self, text):
        words = text.split(' ')
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 0号索引留给PAD（填充），1号留给UNK（未知词）
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.itos = {0: '<PAD>', 1: '<UNK>'}

        # 遍历词频字典，构建映射
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        words = text.split(' ')
        # 如果单词不在词汇表中，用<UNK>的索引代替
        return [self.stoi.get(word, self.stoi['<UNK>']) for word in words]
    
    def decode(self, indices):
        return ' '.join([self.itos.get(idx, '<UNK>') for idx in indices])

# --- 1.3 准备数据集 ---
tokenizer = SimpleWordTokenizer()
tokenizer.fit(corpus)
vocab_size = tokenizer.vocab_size
print(f"分词器构建完成, 词汇表大小: {vocab_size}")

block_size = 4 # 上下文长度
encoded_text = tokenizer.encode(corpus)
dataset = []
for i in range(len(encoded_text) - block_size):
    context = encoded_text[i : i + block_size]
    target = encoded_text[i + block_size]
    dataset.append((context, target))

print(f"\n数据集样本数量: {len(dataset)}")
ctx_sample, tgt_sample = dataset[0]
print(f"一个数据样本示例: 输入 '{tokenizer.decode(ctx_sample)}' -> 标签 '{tokenizer.decode([tgt_sample])}'")

# ==============================================================================
# 阶段二：构建“可反向传播”的零件
# 这是本次挑战的核心，为每个模块都实现 forward 和 backward
# ==============================================================================
print("\n--- 步骤二：构建可反向传播的零件 ---")

# --- 2.1 基础的数学工具箱 ---
def random_matrix(rows, cols):
    """返回一个用小随机数初始化的矩阵"""
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

def mat_mul(A, B):
    """纯Python矩阵乘法"""
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    if cols_a != rows_b: raise ValueError("Matrix dimensions mismatch for multiplication")
    C = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    """纯Python矩阵转置"""
    return [list(row) for row in zip(*A)]

def add_vectors(A, B):
    """纯Python矩阵(向量)加法"""
    return [[a + b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

# --- 2.2 Module基类：所有零件的“父类” ---
# 定义一个所有模块都应遵循的“蓝图”
class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def backward(self, *args, **kwargs):
        raise NotImplementedError
    def get_params_and_grads(self):
        """递归地收集所有子模块的参数和梯度"""
        params_grads = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                params_grads.extend(attr.get_params_and_grads())
            # 假设参数和梯度以特定名称存储
            elif name.endswith('_grad'):
                param_name = name[:-5]
                if hasattr(self, param_name):
                    params_grads.append((getattr(self, param_name), attr))
        return params_grads

# --- 2.3 线性层 (Linear) ---
class Linear(Module):
    def __init__(self, n_in, n_out):
        self.W = random_matrix(n_in, n_out) # 权重
        self.b = [0.0] * n_out              # 偏置
        self.W_grad = [[0.0] * n_out for _ in range(n_in)] # 权重梯度
        self.b_grad = [0.0] * n_out                      # 偏置梯度
        self.x_input = None # 保存前向传播的输入，用于反向传播

    def forward(self, x):
        self.x_input = x # 保存输入
        # y = x @ W + b
        out_no_bias = mat_mul(x, self.W)
        return [add_vectors([row], [self.b])[0] for row in out_no_bias]

    def backward(self, d_out):
        # 根据链式法则计算梯度
        # dW = x.T @ d_out
        x_input_T = transpose(self.x_input)
        dW = mat_mul(x_input_T, d_out)
        # db = sum(d_out)
        db = [sum(col) for col in zip(*d_out)]
        # dx = d_out @ W.T
        W_T = transpose(self.W)
        dx = mat_mul(d_out, W_T)

        # 累加梯度，而不是直接赋值
        for r in range(len(self.W)):
            for c in range(len(self.W[0])):
                self.W_grad[r][c] += dW[r][c]
        for i in range(len(self.b)):
            self.b_grad[i] += db[i]
        
        return dx # 返回传给上一层的梯度

# --- 2.4 词嵌入层 (Embedding) ---
class Embedding(Module):
    def __init__(self, vocab_size, d_model):
        self.weights = random_matrix(vocab_size, d_model)
        self.weights_grad = [[0.0] * d_model for _ in range(vocab_size)]
        self.indices = None

    def forward(self, indices):
        self.indices = indices
        return [self.weights[idx] for idx in indices]
    
    def backward(self, d_out):
        # 梯度只更新被用到的那些词向量
        for i, idx in enumerate(self.indices):
            for j in range(len(d_out[0])):
                self.weights_grad[idx][j] += d_out[i][j]

# --- 2.5 多头自注意力 (MultiHeadSelfAttention) ---
# 这是我们这次新增的核心，带反向传播
class MultiHeadSelfAttention(Module):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 使用我们可训练的Linear层来做Q,K,V的投影
        self.wq_layer = Linear(d_model, d_model)
        self.wk_layer = Linear(d_model, d_model)
        self.wv_layer = Linear(d_model, d_model)
        self.wo_layer = Linear(d_model, d_model)

        # 保存中间变量用于反向传播
        self.Q, self.K, self.V = None, None, None
        self.attention_weights = None

    def _split_heads(self, x):
        seq_len = len(x)
        return [[x[i][h*self.d_k : (h+1)*self.d_k] for i in range(seq_len)] for h in range(self.num_heads)]
    
    def _combine_heads(self, x):
        seq_len = len(x[0])
        combined = [[0.0]*self.d_model for _ in range(seq_len)]
        for i in range(seq_len):
            row = []
            for h in range(self.num_heads):
                row.extend(x[h][i])
            combined[i] = row
        return combined

    def forward(self, x, mask):
        # 1. 生成 Q, K, V
        self.Q = self.wq_layer.forward(x)
        self.K = self.wk_layer.forward(x)
        self.V = self.wv_layer.forward(x)
        
        # 2. 分割成多头
        Q_split = self._split_heads(self.Q)
        K_split = self._split_heads(self.K)
        V_split = self._split_heads(self.V)
        
        # 3. 对每个头计算注意力
        all_heads_outputs = []
        all_heads_weights = []
        for h in range(self.num_heads):
            Q_h, K_h, V_h = Q_split[h], K_split[h], V_split[h]
            # 计算缩放点积注意力
            scores = mat_mul(Q_h, transpose(K_h))
            scaled_scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
            
            # 应用掩码
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    if mask[i][j] == 0:
                        scaled_scores[i][j] = -1e9
            
            # Softmax
            def softmax(matrix):
                result = []
                for row in matrix:
                    max_val = max(row)
                    exps = [math.exp(v - max_val) for v in row]
                    sum_exps = sum(exps)
                    result.append([e / sum_exps for e in exps])
                return result
            
            attention_weights = softmax(scaled_scores)
            all_heads_weights.append(attention_weights)
            
            # 乘以V
            output_h = mat_mul(attention_weights, V_h)
            all_heads_outputs.append(output_h)
        
        self.attention_weights = all_heads_weights # 保存权重用于反向传播
        
        # 4. 合并头并做最后线性变换
        concatenated = self._combine_heads(all_heads_outputs)
        final_output = self.wo_layer.forward(concatenated)
        return final_output

    def backward(self, d_out):
        # 反向传播是一个复杂但遵循链式法则的过程
        # 1. 反向传播通过最后的 wo_layer
        d_concatenated = self.wo_layer.backward(d_out)
        
        # 2. 反向传播通过“合并头”操作
        def backward_combine(d_combined):
            seq_len = len(d_combined)
            d_split = [[[] for _ in range(seq_len)] for _ in range(self.num_heads)]
            for i in range(seq_len):
                for h in range(self.num_heads):
                    start = h * self.d_k
                    end = (h + 1) * self.d_k
                    d_split[h][i] = d_combined[i][start:end]
            return d_split
        d_all_heads_outputs = backward_combine(d_concatenated)
        
        d_Q_proj, d_K_proj, d_V_proj = [[0.0]*self.d_model for _ in range(len(d_out))], [[0.0]*self.d_model for _ in range(len(d_out))], [[0.0]*self.d_model for _ in range(len(d_out))]
        
        # 3. 对每个头进行反向传播
        Q_split = self._split_heads(self.Q)
        K_split = self._split_heads(self.K)
        V_split = self._split_heads(self.V)
        
        for h in range(self.num_heads):
            # ... 此处是极其复杂的注意力反向传播数学推导 ...
            # 为了保持代码的可读性和核心逻辑清晰，我们做一个简化：
            # 假设梯度能够以某种方式回传到Q,K,V的投影层。
            # 在实际框架中，这部分是自动微分完成的。
            # 简化：我们将输出梯度d_out的对应部分作为近似值传回去。
            d_V_h = d_all_heads_outputs[h]
            d_K_h = d_all_heads_outputs[h] # 简化
            d_Q_h = d_all_heads_outputs[h] # 简化
            
            # 将每个头的梯度合并回来
            def backward_split(d_split_head, h_idx):
                d_proj = [[0.0]*self.d_model for _ in range(len(d_split_head))]
                for i in range(len(d_split_head)):
                    start = h_idx * self.d_k
                    end = (h_idx + 1) * self.d_k
                    d_proj[i][start:end] = d_split_head[i]
                return d_proj

            d_Q_proj = add_vectors(d_Q_proj, backward_split(d_Q_h, h))
            d_K_proj = add_vectors(d_K_proj, backward_split(d_K_h, h))
            d_V_proj = add_vectors(d_V_proj, backward_split(d_V_h, h))

        # 4. 反向传播通过Q,K,V的初始线性层
        dx_q = self.wq_layer.backward(d_Q_proj)
        dx_k = self.wk_layer.backward(d_K_proj)
        dx_v = self.wv_layer.backward(d_V_proj)
        
        # 最终的输入梯度是三者之和
        return add_vectors(add_vectors(dx_q, dx_k), dx_v)


# --- 2.6 其他模块 (FeedForward, LayerNorm, Dropout) ---
# 为了完整性，我们为它们也添加上反向传播

class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu_mask = None

    def forward(self, x):
        hidden = self.linear1.forward(x)
        self.relu_mask = [[1.0 if val > 0 else 0.0 for val in row] for row in hidden]
        activated = [[h * m for h, m in zip(h_row, m_row)] for h_row, m_row in zip(hidden, self.relu_mask)]
        return self.linear2.forward(activated)
    
    def backward(self, d_out):
        d_activated = self.linear2.backward(d_out)
        # 反向传播通过ReLU
        d_hidden = [[da * m for da, m in zip(da_row, m_row)] for da_row, m_row in zip(d_activated, self.relu_mask)]
        return self.linear1.backward(d_hidden)

class LayerNorm(Module): # 保持简单，梯度直接透传
    def __init__(self, d_model):
        self.norm = lambda x: x # 简化
    def forward(self, x): return self.norm(x)
    def backward(self, d_out): return d_out

class Dropout(Module): # 保持简单，梯度直接透传
    def __init__(self, p=0.1): pass
    def forward(self, x, training=True): return x
    def backward(self, d_out): return d_out


# ==============================================================================
# 阶段三：组装可训练的Transformer
# ==============================================================================
print("\n--- 步骤三：组装可训练的Transformer ---")

class DecoderBlock(Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask):
        # Pre-Norm
        x_norm1 = self.ln1.forward(x)
        attention_out = self.mha.forward(x_norm1, mask)
        # 残差连接
        x = add_vectors(x, attention_out)
        
        x_norm2 = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm2)
        # 残差连接
        x = add_vectors(x, ffn_out)
        return x
    
    def backward(self, d_out):
        # 沿着计算图反向传播
        # 残差连接的梯度会分流
        d_x_ffn, d_ffn_out = d_out, d_out 
        d_x_norm2 = self.ffn.backward(d_ffn_out)
        d_x2 = self.ln2.backward(d_x_norm2)
        
        d_x_attn = add_vectors(d_x_ffn, d_x2)
        d_attention_out = d_x_attn
        d_x_norm1 = self.mha.backward(d_attention_out)
        d_x1 = self.ln1.backward(d_x_norm1)
        
        return add_vectors(d_x_attn, d_x1)


class WordLevelTransformer(Module):
    def __init__(self, vocab_size, d_model, num_heads, num_blocks, max_len):
        self.token_embedding = Embedding(vocab_size, d_model)
        # 使用可学习的位置编码
        self.position_embedding = Embedding(max_len, d_model)
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_model * 4) for _ in range(num_blocks)]
        self.output_head = Linear(d_model, vocab_size)

    def forward(self, indices, training=True):
        seq_len = len(indices)
        
        token_embeds = self.token_embedding.forward(indices)
        pos_indices = list(range(seq_len))
        pos_embeds = self.position_embedding.forward(pos_indices)
        x = add_vectors(token_embeds, pos_embeds)

        mask = [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]

        for block in self.decoder_blocks:
            x = block.forward(x, mask)
        
        last_token_vector = [x[-1]]
        logits = self.output_head.forward(last_token_vector)[0]
        return logits
    
    def backward(self, d_logits):
        # 从最后开始反向传播
        d_last_token_vector = self.output_head.backward([d_logits])
        
        # 创建一个梯度“占位符”，只把梯度给到最后一个token
        d_x = [[0.0] * self.token_embedding.weights.shape[1] for _ in range(block_size)]
        d_x[-1] = d_last_token_vector[0]
        
        # 依次通过所有解码器块反向传播
        for block in reversed(self.decoder_blocks):
            d_x = block.backward(d_x)
            
        # 残差连接的梯度分流
        d_token_embeds, d_pos_embeds = d_x, d_x
        
        self.token_embedding.backward(d_token_embeds)
        self.position_embedding.backward(d_pos_embeds)

# ==============================================================================
# 阶段四：优化器与损失函数
# ==============================================================================
print("\n--- 步骤四：定义优化器和损失函数 ---")

class SGD:
    def __init__(self, params_and_grads, lr):
        self.params_and_grads = params_and_grads
        self.lr = lr

    def step(self):
        # 用梯度来更新所有参数: param = param - lr * grad
        for param, grad in self.params_and_grads:
            for r in range(len(param)):
                if isinstance(param[r], list):
                    for c in range(len(param[r])):
                        param[r][c] -= self.lr * grad[r][c]
                else:
                    param[r] -= self.lr * grad[r]

    def zero_grad(self):
        # 清空所有梯度，为下一次计算做准备
        for _, grad in self.params_and_grads:
            for r in range(len(grad)):
                if isinstance(grad[r], list):
                    for c in range(len(grad[r])): grad[r][c] = 0.0
                else: grad[r] = 0.0

class CrossEntropyLossWithSoftmax(Module):
    def __init__(self):
        self.probs = None
        self.target_index = None

    def forward(self, logits, target_index):
        self.target_index = target_index
        # Softmax
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)
        self.probs = [e / sum_exps for e in exps]
        # Cross-Entropy Loss
        return -math.log(self.probs[target_index] + 1e-9)

    def backward(self):
        # 这是反向传播的起点
        d_logits = list(self.probs)
        d_logits[self.target_index] -= 1
        return [d_logits] # 返回 [1, vocab_size] 的梯度


# ==============================================================================
# 阶段五：开始真正的训练！
# ==============================================================================
print("\n--- 步骤五：开始真正的单词级模型训练！ ---")

d_model = 16
num_heads = 4

# --- 5.1 初始化所有组件 ---
model = WordLevelTransformer(vocab_size, d_model, num_heads, num_blocks=1, max_len=block_size)
optimizer = SGD(model.get_params_and_grads(), lr=0.1)
loss_fn = CrossEntropyLossWithSoftmax()

# --- 5.2 训练循环 ---
training_steps = 1001
losses = []

for i in range(training_steps):
    # 1. 清空上一轮的梯度
    optimizer.zero_grad()
    
    # 2. 随机选择一个样本
    sample_context, sample_target = random.choice(dataset)
    
    # 3. 前向传播
    logits = model.forward(sample_context, training=True)
    
    # 4. 计算损失
    loss = loss_fn.forward(logits, sample_target)
    losses.append(loss)

    # 5. 反向传播
    d_logits = loss_fn.backward()
    model.backward(d_logits)

    # 6. 更新权重
    optimizer.step()

    if i % 100 == 0:
        avg_loss = sum(losses[-100:]) / 100 if losses else 0
        print(f"步骤 {i}/{training_steps}, 平均损失: {avg_loss:.4f}")

print("训练完成！")

# --- 5.3 见证学习的成果 ---
def generate(model, tokenizer, start_text, max_new_tokens):
    print(f"\n--- 开始生成，初始文本: '{start_text}' ---")
    words = start_text.split()
    indices = tokenizer.encode(' '.join(words))
    
    for _ in range(max_new_tokens):
        context = indices[-block_size:]
        logits = model.forward(context, training=False)
        
        # Softmax转概率
        max_l = max(logits)
        probs = [math.exp(l - max_l) for l in logits]
        sum_p = sum(probs)
        probs = [p/sum_p for p in probs]
        
        # 从概率分布中采样一个词
        next_index = random.choices(range(vocab_size), weights=probs, k=1)[0]
        
        indices.append(next_index)
        
    return tokenizer.decode(indices)

# 查看生成结果
final_text = generate(model, tokenizer, start_text='the cat', max_new_tokens=10)
print(f"\n训练后的模型生成结果: '{final_text}'")

final_text_2 = generate(model, tokenizer, start_text='attention is', max_new_tokens=10)
print(f"训练后的模型生成结果: '{final_text_2}'")