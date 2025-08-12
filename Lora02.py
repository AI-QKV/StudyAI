import math
import random
import json

# ==============================================================================
# 阶段一：基础架构 - 保持稳定、健壮的底层工具
# ==============================================================================
print("--- 步骤一：定义基础架构 ---")

def random_matrix(rows, cols):
    """返回一个用Xavier/Glorot初始化思想的随机矩阵，有助于稳定训练。"""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def mat_mul(A, B):
    """纯Python矩阵乘法"""
    if not A or not A[0] or not B or not B[0]: return []
    rows_a, cols_a, rows_b, cols_b = len(A), len(A[0]), len(B), len(B[0])
    if cols_a != rows_b: raise ValueError(f"矩阵维度不匹配: {cols_a} != {rows_b}")
    C = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    """纯Python矩阵转置"""
    if not A or not A[0]: return []
    return [list(row) for row in zip(*A)]

def add_vectors(A, B):
    """纯Python矩阵(向量)加法"""
    return [[a + b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

class Module:
    """所有模型组件的父类，实现了自动化的梯度和参数管理。"""
    def forward(self, *args, **kwargs): raise NotImplementedError
    def backward(self, *args, **kwargs): raise NotImplementedError
    
    def get_params_and_grads(self):
        params_grads = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                params_grads.extend(attr.get_params_and_grads())
            elif name.endswith('_grad'):
                param_name = name[:-5]
                if hasattr(self, param_name):
                    if getattr(self, param_name + "_trainable", True):
                         params_grads.append((getattr(self, param_name), attr))
        return params_grads

    def zero_grad(self):
        for param, grad in self.get_params_and_grads():
            if not grad: continue
            for r in range(len(grad)):
                if isinstance(grad[r], list):
                    for c in range(len(grad[r])): grad[r][c] = 0.0
                else:
                    grad[r] = 0.0

class Dropout(Module):
    """一个功能完整的Dropout层"""
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None
        self.is_training = True # 默认是训练模式

    def forward(self, x):
        if self.is_training:
            # 创建一个与x形状相同的掩码，以概率p将元素置为0
            self.mask = [[1.0 if random.random() > self.p else 0.0 for _ in row] for row in x]
            # 应用掩码并进行缩放(inverted dropout)，以保持期望值不变
            # 这样在推理时就无需做任何改动
            res = []
            for r in range(len(x)):
                row = []
                for c in range(len(x[0])):
                    # 只有当p<1时才进行缩放，防止除以0
                    scale = 1.0 / (1.0 - self.p) if self.p < 1.0 else 0.0
                    row.append(x[r][c] * self.mask[r][c] * scale)
                res.append(row)
            return res
        else:
            # 在评估/推理模式下，Dropout层什么都不做
            return x

    def backward(self, d_out):
        # 梯度也需要通过同样的掩码进行屏蔽
        if not self.is_training:
            return d_out
        res = []
        for r in range(len(d_out)):
            row = []
            for c in range(len(d_out[0])):
                scale = 1.0 / (1.0 - self.p) if self.p < 1.0 else 0.0
                row.append(d_out[r][c] * self.mask[r][c] * scale)
            res.append(row)
        return res



# ==============================================================================
# 阶段二：数据准备
# ==============================================================================
print("\n--- 步骤二：创建数据集和单词分词器 ---")
word_list = [
    'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', '.',
    'a', 'cat', 'sat', 'on', 'mat', 'and', 'played', 'with', 'ball', '.',
    'transformer', 'models', 'are', 'powerful', 'for', 'nlp', 'tasks', '.',
    'attention', 'is', 'all', 'you', 'need', 'is', 'a', 'famous', 'paper', '.',
    'we', 'are', 'building', 'this', 'model', 'from', 'scratch', 'to', 'learn', '.'
]
corpus = ' '.join(random.choices(word_list, k=300))
class SimpleWordTokenizer:
    def __init__(self): self.stoi, self.itos = {}, {}
    def fit(self, text):
        words = sorted(list(set(text.split(' '))))
        self.stoi = {word: i for i, word in enumerate(words)}
        self.itos = {i: word for i, word in enumerate(words)}
        self.vocab_size = len(words)
    def encode(self, text): return [self.stoi.get(word, -1) for word in text.split(' ')]
    def decode(self, indices): return ' '.join([self.itos.get(idx, '<UNK>') for idx in indices])

tokenizer = SimpleWordTokenizer()
tokenizer.fit(corpus)
vocab_size = tokenizer.vocab_size
print(f"分词器构建完成, 词汇表大小: {vocab_size}")
block_size = 5
encoded_text = tokenizer.encode(corpus)
dataset = [(encoded_text[i:i+block_size], encoded_text[i+block_size]) for i in range(len(encoded_text) - block_size)]
print(f"数据集样本数量: {len(dataset)}")

# ==============================================================================
# 阶段三：LoRA核心实现 - 改造Linear层
# ==============================================================================
print("\n--- 步骤三：注入LoRA灵魂 - 改造Linear层 ---")

class LoRALinear(Module):
    def __init__(self, n_in, n_out, rank, alpha=1.0):
        self.W, self.b = random_matrix(n_in, n_out), [0.0] * n_out
        self.W_trainable, self.b_trainable = True, True
        self.W_grad, self.b_grad = [[0.0] * n_out for _ in range(n_in)], [0.0] * n_out
        
        self.lora_A = random_matrix(n_in, rank)
        self.lora_B = [[0.0] * n_out for _ in range(rank)]
        self.lora_A_trainable, self.lora_B_trainable = False, False
        self.lora_A_grad = [[0.0] * rank for _ in range(n_in)]
        self.lora_B_grad = [[0.0] * n_out for _ in range(rank)]

        self.rank, self.alpha = rank, alpha
        self.x_input, self.lora_A_output = None, None

    def enable_lora(self):
        self.W_trainable, self.b_trainable = False, False
        self.lora_A_trainable, self.lora_B_trainable = True, True
        print(f"  > LoRALinear层 ({len(self.W)}x{len(self.W[0])}) 已激活LoRA模式!")

    def forward(self, x):
        self.x_input = x
        base_path_output = [add_vectors([row], [self.b])[0] for row in mat_mul(x, self.W)]
        if self.lora_A_trainable:
            self.lora_A_output = mat_mul(x, self.lora_A)
            lora_B_output = mat_mul(self.lora_A_output, self.lora_B)
            scale = self.alpha / self.rank
            lora_path_output = [[val * scale for val in row] for row in lora_B_output]
            return add_vectors(base_path_output, lora_path_output)
        else:
            return base_path_output

    def backward(self, d_out):
        dx_base = mat_mul(d_out, transpose(self.W))
        dx_lora = [[0.0] * len(self.W) for _ in range(len(d_out))]
        if self.lora_A_trainable:
            scale = self.alpha / self.rank
            d_lora_path = [[val * scale for val in row] for row in d_out]
            lora_A_output_T = transpose(self.lora_A_output)
            dB = mat_mul(lora_A_output_T, d_lora_path)
            d_lora_A_output = mat_mul(d_lora_path, transpose(self.lora_B))
            dA = mat_mul(transpose(self.x_input), d_lora_A_output)
            dx_lora = mat_mul(d_lora_A_output, transpose(self.lora_A))
            for r in range(self.rank):
                for c in range(len(self.lora_B[0])): self.lora_B_grad[r][c] += dB[r][c]
            for r in range(len(self.lora_A)):
                for c in range(self.rank): self.lora_A_grad[r][c] += dA[r][c]
        if self.W_trainable:
            x_input_T = transpose(self.x_input)
            dW = mat_mul(x_input_T, d_out)
            db = [sum(col) for col in zip(*d_out)]
            for r in range(len(self.W)):
                for c in range(len(self.W[0])): self.W_grad[r][c] += dW[r][c]
            for i in range(len(self.b)): self.b_grad[i] += db[i]
        return add_vectors(dx_base, dx_lora)

# ==============================================================================
# 阶段四：组装一个带LoRA接口的、结构正确的Transformer
# ==============================================================================
print("\n--- 步骤四：组装带LoRA接口的、结构正确的Transformer ---")

class Embedding(Module): # (与之前版本相同)
    def __init__(self, vocab_size, d_model):
        self.weights, self.weights_grad = random_matrix(vocab_size, d_model), [[0.0] * d_model for _ in range(vocab_size)]
        self.indices = None
    def forward(self, indices):
        self.indices = indices
        return [self.weights[idx] for idx in indices]
    def backward(self, d_out):
        for i, idx in enumerate(self.indices):
            for j in range(len(d_out[0])): self.weights_grad[idx][j] += d_out[i][j]

class RMSNorm(Module): # (简化版，保持不变)
    def __init__(self, d_model): self.d_model = d_model
    def forward(self, x): return x
    def backward(self, d_out): return d_out

# 关键修正：FFN应该是一个独立的模块，包含两个线性层
class FeedForward(Module):
    def __init__(self, d_model, d_ff, lora_rank):
        self.linear1 = LoRALinear(d_model, d_ff, rank=lora_rank)
        self.linear2 = LoRALinear(d_ff, d_model, rank=lora_rank)
        self.relu_mask = None
    def forward(self, x):
        hidden = self.linear1.forward(x)
        self.relu_mask = [[1.0 if val > 0 else 0.0 for val in row] for row in hidden]
        activated = [[h * m for h, m in zip(h_row, m_row)] for h_row, m_row in zip(hidden, self.relu_mask)]
        return self.linear2.forward(activated)
    def backward(self, d_out):
        d_activated = self.linear2.backward(d_out)
        d_hidden = [[da * m for da, m in zip(da_row, m_row)] for da_row, m_row in zip(d_activated, self.relu_mask)]
        return self.linear1.backward(d_hidden)

class StandardDecoderBlock(Module):
    def __init__(self, d_model, num_heads, dropout_p=0.1, lora_rank=4):
        # 简化版注意力，但FFN是完整的
        self.attn = LoRALinear(d_model, d_model, rank=lora_rank) 
        self.ffn = FeedForward(d_model, d_model * 4, lora_rank=lora_rank)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.dropout = Dropout(dropout_p)

    def forward(self, x, training=True):
        self.dropout.is_training = training
        attn_out = self.attn.forward(self.ln1.forward(x))
        x = add_vectors(x, self.dropout.forward(attn_out))
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = add_vectors(x, ffn_out)
        return x

    def backward(self, d_out):
        d_add2 = d_out
        d_x_from_ffn, d_ffn_out = d_add2, d_add2
        d_ln2_out = self.ffn.backward(d_ffn_out)
        d_add1_from_ln2 = self.ln2.backward(d_ln2_out)
        d_add1 = add_vectors(d_x_from_ffn, d_add1_from_ln2)
        d_x_from_attn, d_attn_out_path = d_add1, d_add1
        d_attn_out = self.dropout.backward(d_attn_out_path)
        d_ln1_out = self.attn.backward(d_attn_out)
        d_x_from_ln1 = self.ln1.backward(d_ln1_out)
        return add_vectors(d_x_from_attn, d_x_from_ln1)

class LoraTransformer(Module):
    def __init__(self, vocab_size, d_model, num_blocks, max_len, lora_rank=4):
        self.d_model = d_model # 保存d_model以修复bug
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(max_len, d_model)
        self.decoder_blocks = [StandardDecoderBlock(d_model, num_heads=0, lora_rank=lora_rank) for _ in range(num_blocks)]
        self.output_head = LoRALinear(d_model, vocab_size, rank=lora_rank)

    def forward(self, indices, training=True):
        x = add_vectors(self.token_embedding.forward(indices), 
                        self.position_embedding.forward(list(range(len(indices)))))
        for block in self.decoder_blocks:
            x = block.forward(x, training)
        return self.output_head.forward([x[-1]])[0]

    def backward(self, d_logits):
        # 传入的d_logits是1D列表
        d_last_token = self.output_head.backward([d_logits]) # backward需要2D列表
        
        # --- 关键BUG修复 ---
        # 之前的代码: d_x = [[0.0] * self.output_head.W[0].__len__() ... ] -> 宽度为40 (vocab_size)
        # 正确的代码: d_x的宽度应该是d_model (32)，因为这是模型内部流转的维度
        d_x = [[0.0] * self.d_model for _ in range(block_size)]
        d_x[-1] = d_last_token[0] # d_last_token是[[...]]，所以取第一个元素
        
        for block in reversed(self.decoder_blocks):
            d_x = block.backward(d_x)
        self.token_embedding.backward(d_x)
        self.position_embedding.backward(d_x)
        
    def enable_lora_finetuning(self):
        print("\n--- 激活LoRA微调模式 ---")
        # 遍历所有属性，找到LoRALinear实例并激活lora
        for name, module in self.__dict__.items():
            if isinstance(module, LoRALinear):
                module.enable_lora()
            if isinstance(module, list) and all(isinstance(m, Module) for m in module):
                 for m in module:
                     if hasattr(m, 'enable_lora_finetuning'):
                         m.enable_lora_finetuning()
            if isinstance(module, Module):
                 if hasattr(module, 'enable_lora_finetuning'):
                         module.enable_lora_finetuning()

# ==============================================================================
# 阶段五：模型保存/加载 与 完整的训练流程
# ==============================================================================
print("\n--- 步骤五：实现模型保存/加载与完整训练流程 ---")

def save_weights(model, filename):
    weights_to_save, param_id = {}, 0
    for param, _ in model.get_params_and_grads():
        weights_to_save[f'param_{param_id}'] = param
        param_id += 1
    with open(filename, 'w') as f: json.dump(weights_to_save, f)
    print(f"权重已保存到 {filename}")

def load_weights(model, filename):
    with open(filename, 'r') as f: weights_from_file = json.load(f)
    param_id = 0
    for param, _ in model.get_params_and_grads():
        key = f'param_{param_id}'
        if key in weights_from_file:
            loaded_param = weights_from_file[key]
            for r in range(len(param)):
                if isinstance(param[r], list):
                    for c in range(len(param[r])): param[r][c] = loaded_param[r][c]
                else: param[r] = loaded_param[r]
        param_id += 1
    print(f"已从 {filename} 加载权重")

class SGD:
    def __init__(self, model, lr): self.model, self.lr = model, lr
    def step(self):
        for param, grad in self.model.get_params_and_grads():
            for r in range(len(param)):
                if isinstance(param[r], list):
                    for c in range(len(param[r])): param[r][c] -= self.lr * grad[r][c]
                else: param[r] -= self.lr * grad[r]
class CrossEntropyLossWithSoftmax:
    def __init__(self): self.probs, self.target_index = None, None
    def forward(self, logits, target_index):
        self.target_index, max_logit = target_index, max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)
        self.probs = [e / sum_exps for e in exps]
        return -math.log(self.probs[target_index] + 1e-9)
    def backward(self):
        d_logits = list(self.probs)
        d_logits[self.target_index] -= 1
        return d_logits

# --- 5.1 "预训练"一个基础模型 ---
print("\n--- 5.1 开始“预训练”基础模型 ---")
d_model, num_blocks, lora_rank = 32, 2, 4
base_model = LoraTransformer(vocab_size, d_model, num_blocks, block_size, lora_rank)
optimizer = SGD(base_model, lr=0.1)
loss_fn = CrossEntropyLossWithSoftmax()
losses = []

for i in range(501):
    base_model.zero_grad()
    ctx, tgt = random.choice(dataset)
    logits = base_model.forward(ctx)
    loss = loss_fn.forward(logits, tgt)
    losses.append(loss)
    d_logits = loss_fn.backward()
    base_model.backward(d_logits)
    optimizer.step()
    if i % 100 == 0:
        # 修复平均损失计算bug
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        print(f"  预训练步骤 {i}, 最近损失: {avg_loss:.4f}")

# --- 5.2 保存基础模型权重 ---
save_weights(base_model, "base_model_weights.json")

# --- 5.3 开始LoRA微调 ---
print("\n--- 5.2 开始LoRA微调 ---")
lora_model = LoraTransformer(vocab_size, d_model, num_blocks, block_size, lora_rank)
load_weights(lora_model, "base_model_weights.json")
lora_model.enable_lora_finetuning()
lora_optimizer = SGD(lora_model, lr=0.05) # 使用一个更小的学习率进行微调
print(f"LoRA微调将只训练 {len(lora_model.get_params_and_grads())} 个参数张量。")
losses = []

for i in range(501):
    lora_model.zero_grad()
    ctx, tgt = random.choice(dataset)
    logits = lora_model.forward(ctx)
    loss = loss_fn.forward(logits, tgt)
    losses.append(loss)
    d_logits = loss_fn.backward()
    lora_model.backward(d_logits)
    lora_optimizer.step()
    if i % 100 == 0:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        print(f"  LoRA微调步骤 {i}, 最近损失: {avg_loss:.4f}")

# --- 5.4 保存LoRA适配器权重 ---
class LoraAdapters(Module):
    def __init__(self, model):
        self.lora_params = model.get_params_and_grads()
    def get_params_and_grads(self):
        return self.lora_params
lora_adapters = LoraAdapters(lora_model)
save_weights(lora_adapters, "lora_adapters.json")

# ==============================================================================
# 阶段六：加载模型与适配器进行最终推理
# ==============================================================================
print("\n--- 步骤六：加载模型与适配器进行最终推理 ---")
final_model = LoraTransformer(vocab_size, d_model, num_blocks, block_size, lora_rank)
print("1. 加载预训练的基础权重...")
load_weights(final_model, "base_model_weights.json")
final_model.enable_lora_finetuning()
print("2. 加载训练好的LoRA适配器权重...")
adapters_to_load = LoraAdapters(final_model)
load_weights(adapters_to_load, "lora_adapters.json")
print("\n模型和LoRA适配器加载并合并完成！")



if __name__ == "__main__":
    def generate(model, tokenizer, start_text, max_new_tokens):
        print(f"\n--- 开始生成，初始文本: '{start_text}' ---")
        indices = tokenizer.encode(start_text)
        for _ in range(max_new_tokens):
            context = indices[-block_size:]
            logits = model.forward(context, training=False)
            max_l = max(logits)
            probs = [math.exp(l - max_l) for l in logits]
            sum_p = sum(probs)
            probs = [p/sum_p for p in probs]
            next_index = random.choices(range(vocab_size), weights=probs, k=1)[0]
            indices.append(next_index)
        return tokenizer.decode(indices)


    input_word = input("请输入提示词（Q表示退出）:")
    while input_word != "Q":    
        final_text = generate(final_model, tokenizer, start_text=input_word, max_new_tokens=10)
        print(f"\n微调后的模型生成结果: '{final_text}'")
        input_word = input("请输入提示词（Q表示退出）:")