import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
import time
import os
import json
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size:int=100277; context_length:int=512; hidden_size:int=1536
    num_layers:int=24; num_heads:int=16; ffn_hidden_size:int=6144; norm_eps:float=1e-5

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x): return self.weight*x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate=nn.Linear(c.hidden_size,c.ffn_hidden_size,bias=False)
        self.up=nn.Linear(c.hidden_size,c.ffn_hidden_size,bias=False)
        self.down=nn.Linear(c.ffn_hidden_size,c.hidden_size,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class LoRALinear(nn.Module):
    def __init__(self,linear,rank=16,alpha=32):
        super().__init__()
        self.linear=linear
        self.linear.weight.requires_grad_(False)
        d_out,d_in=linear.weight.shape
        self.lora_A=nn.Parameter(torch.randn(rank,d_in)*0.01)
        self.lora_B=nn.Parameter(torch.zeros(d_out,rank))
        self.scale=alpha/rank
    def forward(self,x):
        return self.linear(x)+(x@self.lora_A.T@self.lora_B.T)*self.scale

class Attention(nn.Module):
    def __init__(self,c,use_lora=True):
        super().__init__(); self.h=c.num_heads; self.d=c.hidden_size//c.num_heads
        q_linear=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        v_linear=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.q=LoRALinear(q_linear) if use_lora else q_linear
        self.k=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.v=LoRALinear(v_linear) if use_lora else v_linear
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

class KalaiNovaAssistant(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.embed=nn.Embedding(c.vocab_size,c.hidden_size)
        self.layers=nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm=RMSNorm(c.hidden_size)
        self.head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        self.head.weight=self.embed.weight
    def forward(self,x,targets=None,mask=None):
        x=self.embed(x)
        for layer in self.layers:
            if self.training: x=grad_checkpoint(layer,x,use_reentrant=False)
            else: x=layer(x)
        logits=self.head(self.norm(x))
        loss=None
        if targets is not None:
            loss_all=F.cross_entropy(logits.view(-1,self.config.vocab_size),targets.view(-1),reduction='none')
            if mask is not None:
                loss=(loss_all*mask.view(-1)).sum()/mask.sum().clamp(min=1)
            else:
                loss=loss_all.mean()
        return logits,loss
    def count_params(self): return sum(p.numel() for p in self.parameters())
    def count_trainable(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

enc=tiktoken.get_encoding("cl100k_base")
ANSWER_START=enc.encode("### Answer\n")

def build_batch(qa_pairs,context,batch_size,device):
    xs,ys,masks=[],[],[]
    for _ in range(batch_size):
        idx=torch.randint(0,len(qa_pairs),(1,)).item()
        q,a=qa_pairs[idx]
        text="### Question\n"+q+"\n\n### Answer\n"+a+"\n\n"
        ids=enc.encode(text)[:context+1]
        if len(ids)<context+1: ids=ids+[0]*(context+1-len(ids))
        x=ids[:context]; y=ids[1:context+1]
        mask=[0]*context
        for i in range(len(x)-len(ANSWER_START)):
            if x[i:i+len(ANSWER_START)]==ANSWER_START:
                for j in range(i+len(ANSWER_START),context): mask[j]=1
                break
        xs.append(x); ys.append(y); masks.append(mask)
    return (torch.tensor(xs,dtype=torch.long).to(device),
            torch.tensor(ys,dtype=torch.long).to(device),
            torch.tensor(masks,dtype=torch.float).to(device))

# Load Grok Q&A data
print("Loading Grok Q&A data...")
with open('grok_flutter_qa.txt','r') as f: content=f.read()
qa_pairs=[]
for block in content.split('### Question\n')[1:]:
    if '### Answer\n' in block:
        parts=block.split('### Answer\n')
        q=parts[0].strip(); a=parts[1].split('### Question')[0].strip()
        if q and a and len(a)>20: qa_pairs.append((q,a))
print(f"Loaded {len(qa_pairs)} Q&A pairs")

device=torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {device}")

config=ModelConfig()
model=KalaiNovaAssistant(config).to(device)

# Load our trained checkpoint
ckpt_path='checkpoints/kalainova_best.pt'
if not os.path.exists(ckpt_path):
    ckpt_path='checkpoints/kalainova_clean_final.pt'
if not os.path.exists(ckpt_path):
    ckpt_path='checkpoints/instruct_final.pt'

print(f"Loading checkpoint: {ckpt_path}")
ckpt=torch.load(ckpt_path,map_location=device)
model.load_state_dict(ckpt['model_state'],strict=False)
print(f"Loaded! Total params: {model.count_params()/1e6:.1f}M")

# Freeze everything except LoRA
for name,param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad_(False)

print(f"Trainable (LoRA): {model.count_trainable()/1e6:.2f}M params")

optimizer=torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-4,weight_decay=0.01
)

os.makedirs('checkpoints',exist_ok=True)
context,batch,max_steps=512,2,1000
print(f"\nLoRA fine-tuning {max_steps} steps on Grok data...")
print("="*50)

model.train(); start=time.time()
for step in range(1,max_steps+1):
    x,y,mask=build_batch(qa_pairs,context,batch,device)
    optimizer.zero_grad()
    _,loss=model(x,y,mask=mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
    optimizer.step()
    if step%50==0:
        elapsed=time.time()-start
        print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | {step/elapsed:.2f} steps/s")
    if step%500==0:
        torch.save({'model_state':model.state_dict(),'step':step},f"checkpoints/lora_{step}.pt")
        print(f"Saved: lora_{step}.pt")

torch.save({'model_state':model.state_dict(),'step':max_steps},"checkpoints/kalainova_lora_final.pt")
print("\nDone! Testing...")

# Quick test
model.eval()
stop_tokens=enc.encode('\n\n### Question')
def generate(prompt,max_new_tokens=150,temperature=0.7,top_k=40):
    ids=enc.encode(prompt)
    x=torch.tensor([ids],dtype=torch.long).to(device)
    generated=[]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits,_=model(x)
            logits=logits[0,-1,:]/temperature
            top_vals,top_idx=torch.topk(logits,top_k)
            probs=F.softmax(top_vals,dim=-1)
            next_tok=top_idx[torch.multinomial(probs,1)].item()
            generated.append(next_tok)
            x=torch.cat([x,torch.tensor([[next_tok]]).to(device)],dim=1)
            if len(generated)>=len(stop_tokens):
                if generated[-len(stop_tokens):]==stop_tokens: break
            if len(generated)>=20:
                last5=generated[-5:]
                prev=generated[-25:-5]
                repeats=sum(1 for i in range(0,len(prev)-4,5) if prev[i:i+5]==last5)
                if repeats>=3: break
    full=enc.decode(ids+generated)
    return full.split('### Answer\n')[-1].split('### Question')[0].strip()

tests=['What is StatefulWidget in Flutter?','How do I use setState in Flutter?','What is hot reload in Flutter?']
for q in tests:
    print(f"\nQ: {q}")
    prompt = '### Question\n' + q + '\n\n### Answer\n'
    print('A:', generate(prompt))
