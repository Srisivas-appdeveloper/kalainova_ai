"""
KalaiNova Coder - Distributed Training
COORDINATOR - MacBook M4

Flow:
1. Start DHT and averager
2. Wait for HP to connect
3. Confirm connection - ask user to start
4. Both train together, sync every 10 minutes
"""

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import torch, torch.nn as nn, torch.nn.functional as F
import tiktoken, hivemind, time, os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size:int=100277; context_length:int=256; hidden_size:int=1024
    num_layers:int=12; num_heads:int=16; ffn_hidden_size:int=4096; norm_eps:float=1e-5

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x):
        return self.weight*x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate=nn.Linear(c.hidden_size,c.ffn_hidden_size,bias=False)
        self.up=nn.Linear(c.hidden_size,c.ffn_hidden_size,bias=False)
        self.down=nn.Linear(c.ffn_hidden_size,c.hidden_size,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.h=c.num_heads; self.d=c.hidden_size//c.num_heads
        self.q=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.k=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.v=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.o=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
    def forward(self,x):
        B,T,C=x.shape
        q=self.q(x).view(B,T,self.h,self.d).transpose(1,2)
        k=self.k(x).view(B,T,self.h,self.d).transpose(1,2)
        v=self.v(x).view(B,T,self.h,self.d).transpose(1,2)
        return self.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.n1=RMSNorm(c.hidden_size); self.attn=Attention(c)
        self.n2=RMSNorm(c.hidden_size); self.ffn=SwiGLU(c)
    def forward(self,x): x=x+self.attn(self.n1(x)); x=x+self.ffn(self.n2(x)); return x

class KalaiNovaCoder(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.embed=nn.Embedding(c.vocab_size,c.hidden_size)
        self.layers=nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm=RMSNorm(c.hidden_size)
        self.head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        self.head.weight=self.embed.weight
    def forward(self,x,targets=None):
        x=self.embed(x)
        for l in self.layers: x=l(x)
        logits=self.head(self.norm(x))
        loss=F.cross_entropy(logits.view(-1,self.config.vocab_size),targets.view(-1)) if targets is not None else None
        return logits,loss
    def count_params(self): return sum(p.numel() for p in self.parameters())


print('='*50)
print('KalaiNova Coder - COORDINATOR (MacBook M4)')
print('='*50)

# STEP 1: Start permanent DHT
dht = hivemind.DHT(
    host_maddrs=['/ip4/0.0.0.0/tcp/59640'],
    announce_maddrs=['/ip4/192.168.0.2/tcp/59640'],
    start=True
)
my_id = str(dht.peer_id)
print(f'\nNode ID: {my_id}')
print(f'Address: /ip4/192.168.0.2/tcp/59640/p2p/{my_id}')
print('\n*** Share this address with HP Worker ***\n')

# STEP 2: Start averager BEFORE any CUDA
dummy = torch.randn(10)
averager = hivemind.DecentralizedAverager(
    averaged_tensors=[dummy],
    dht=dht,
    prefix='kalainova_handshake',
    target_group_size=2,
    start=True
)
print('Averager ready. Waiting for HP to connect...')
print('(Run hp_worker.py on HP laptop now)\n')

# STEP 3: Wait for HP to join - check peer count
def get_peer_count():
    try:
        peers = dht.get_visible_maddrs()
        # Filter out local addresses
        external = [p for p in peers if '127.0.0.1' not in str(p)]
        return len(external)
    except:
        return 0

hp_connected = False
while not hp_connected:
    time.sleep(3)
    # Try averaging - if it succeeds, HP is connected
    try:
        averager.step(timeout=3.0)
        hp_connected = True
        print('✅ HP Laptop CONNECTED!')
    except:
        print('⏳ Waiting for HP laptop...')

# STEP 4: Ask user to confirm start
print('\n' + '='*50)
answer = input('HP is connected. Shall I start training? (y/n): ')
if answer.lower() != 'y':
    print('Training cancelled.')
    exit()

# Signal HP to start (store in DHT)
dht.store('training_start', b'1', expiration_time=hivemind.get_dht_time() + 3600)
print('\nSignal sent to HP. Starting training...')
print('='*50)

# STEP 5: Load data and model
with open('coding_data.txt', 'r') as f: text = f.read()
enc = tiktoken.get_encoding('cl100k_base')
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
print(f'Tokens: {len(tokens):,}')

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')
config = ModelConfig()
model = KalaiNovaCoder(config).to(device)
print(f'Parameters: {model.count_params()/1e6:.1f}M')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
cpu_params = [p.data.cpu().clone() for p in model.parameters()]

# Update averager with real params
averager2 = hivemind.DecentralizedAverager(
    averaged_tensors=cpu_params,
    dht=dht,
    prefix='kalainova_weights',
    target_group_size=2,
    start=True
)

# STEP 6: Train
os.makedirs('checkpoints', exist_ok=True)
context, batch, max_steps = 256, 4, 5000
model.train()
train_start = time.time()
last_sync = time.time()
step = 0

print('\n[MAC] Training started!')
print('='*50)

while step < max_steps:
    step += 1
    idx = torch.randint(0, len(tokens)-context-1, (batch,))
    x = torch.stack([tokens[i:i+context] for i in idx]).to(device)
    y = torch.stack([tokens[i+1:i+context+1] for i in idx]).to(device)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        print(f'[MAC] Step {step}/{max_steps} | Loss: {loss.item():.4f} | Time: {time.time()-train_start:.0f}s')

    # Time-based sync every 10 minutes
    now = time.time()
    if now - last_sync >= 600:
        print(f'\n[MAC] ⏰ Syncing weights with HP at step {step}...')
        for cp, mp in zip(cpu_params, model.parameters()):
            cp.copy_(mp.data.cpu())
        try:
            averager2.step(timeout=60.0)
            for cp, mp in zip(cpu_params, model.parameters()):
                mp.data.copy_(cp.to(device))
            print(f'[MAC] ✅ SYNCED WITH HP!\n')
        except Exception as e:
            print(f'[MAC] ⚠️ Sync failed: {e}\n')
        last_sync = now

    if step % 1000 == 0:
        torch.save({'model_state': model.state_dict(), 'step': step},
                   f'checkpoints/mac_{step}.pt')
        print(f'[MAC] Checkpoint saved at step {step}')

print('[MAC] Training Complete!')