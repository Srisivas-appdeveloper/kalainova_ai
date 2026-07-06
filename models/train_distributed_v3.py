
import torch, torch.nn as nn, torch.nn.functional as F, tiktoken, hivemind, time, os
from dataclasses import dataclass

COORDINATOR_ADDR = ""
IS_COORDINATOR = COORDINATOR_ADDR == ""

def get_device():
    if IS_COORDINATOR and torch.backends.mps.is_available():
        print("Using Apple Metal (M4 GPU)")
        return torch.device("mps")
    if not IS_COORDINATOR and torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

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

print("="*50)
print("KalaiNova - Distributed Training")
print("Role: COORDINATOR (MacBook M4)")
print("="*50)

dht = hivemind.DHT(host_maddrs=["/ip4/0.0.0.0/tcp/59640"], start=True)
print("Node ID:", dht.peer_id)
for addr in dht.get_visible_maddrs(): print("Address:", addr)
print("\n*** COPY ADDRESS ABOVE TO HP WORKER ***\n")

with open("coding_data.txt","r") as f: text=f.read()
enc=tiktoken.get_encoding("cl100k_base")
tokens=torch.tensor(enc.encode(text),dtype=torch.long)
print(f"Tokens: {len(tokens):,}")

device=get_device()
config=ModelConfig()
model=KalaiNovaCoder(config).to(device)
print(f"Parameters: {model.count_params()/1e6:.1f}M")

optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1)

averager=hivemind.DecentralizedAverager(
    parameters=list(model.parameters()),
    dht=dht, prefix="kalainova_weights",
    target_group_size=2, averaging_expiry=15.0, start=True,
)

os.makedirs("checkpoints",exist_ok=True)
context,batch,max_steps,avg_every=256,4,5000,100
print(f"Training started! Averaging every {avg_every} steps")
print("="*50)

model.train(); start=time.time()
for step in range(1,max_steps+1):
    idx=torch.randint(0,len(tokens)-context-1,(batch,))
    x=torch.stack([tokens[i:i+context] for i in idx]).to(device)
    y=torch.stack([tokens[i+1:i+context+1] for i in idx]).to(device)
    optimizer.zero_grad(); _,loss=model(x,y); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    if step%avg_every==0:
        try: averager.step(timeout=10.0); print(f"✅ Step {step} weights averaged!")
        except Exception as e: print(f"⚠️ Step {step} avg skipped: {e}")
        print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | Time: {time.time()-start:.1f}s")
    if step%1000==0:
        torch.save({"model_state":model.state_dict(),"config":config,"step":step},f"checkpoints/dist_{step}.pt")
        print(f"Checkpoint saved!")
print("Training Complete!")
