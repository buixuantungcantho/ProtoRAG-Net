from __future__ import annotations
import hashlib, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

NUM_LABELS = 7
NUM_EVIDENCE = 26

class ImageEncoder(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, out_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_dim = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(in_dim,out_dim), nn.LayerNorm(out_dim), nn.GELU())
    def forward(self, x):
        return self.proj(self.backbone(x))

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim,dim//2),nn.Tanh(),nn.Linear(dim//2,1))
    def forward(self, x, mask):
        s=self.score(x).squeeze(-1)
        s=s.masked_fill(~mask, torch.finfo(s.dtype).min)
        a=torch.softmax(s,dim=1)
        z=(x*a.unsqueeze(-1)).sum(dim=1)
        return z,a

class BaseMIL(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dim=256, attention=True):
        super().__init__()
        self.encoder=ImageEncoder(backbone,pretrained,hidden_dim)
        self.attention=attention
        self.pool=AttentionPool(hidden_dim) if attention else None
        self.cls=nn.Linear(hidden_dim,NUM_LABELS)
    def encode_views(self, images):
        b,n,c,h,w=images.shape
        return self.encoder(images.reshape(b*n,c,h,w)).reshape(b,n,-1)
    def forward(self, images, view_mask, **kwargs):
        x=self.encode_views(images)
        if self.attention:
            z,a=self.pool(x,view_mask)
        else:
            m=view_mask.unsqueeze(-1).float()
            z=(x*m).sum(1)/m.sum(1).clamp_min(1)
            a=view_mask.float()/view_mask.float().sum(1,keepdim=True).clamp_min(1)
        return {"logits":self.cls(z),"embedding":z,"view_attention":a}

class ClinicalTextEncoder(nn.Module):
    """Content-aware report encoder with deterministic hashed token IDs.

    This avoids metadata-only text statistics and avoids external model downloads.
    Tokens are whitespace/punctuation segmented, deterministically hashed, embedded,
    then encoded by a bidirectional GRU. Text is used only for the privileged teacher.
    """
    def __init__(self, out_dim=256, vocab_size=16384, embedding_dim=192, max_tokens=256, bidirectional=True):
        super().__init__()
        self.vocab_size=int(vocab_size); self.max_tokens=int(max_tokens)
        self.emb=nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        h=max(64,out_dim//2) if bidirectional else out_dim
        self.gru=nn.GRU(embedding_dim,h,batch_first=True,bidirectional=bidirectional)
        gru_dim=h*(2 if bidirectional else 1)
        self.proj=nn.Sequential(nn.Linear(gru_dim,out_dim),nn.LayerNorm(out_dim),nn.GELU())
    @staticmethod
    def _tokens(text):
        return re.findall(r"\w+|[^\w\s]", (text or "").lower(), flags=re.UNICODE)
    def _id(self, tok):
        h=hashlib.blake2b(tok.encode('utf-8'),digest_size=8).digest()
        return 1 + int.from_bytes(h,'little') % (self.vocab_size-1)
    def encode_ids(self,texts,device):
        seqs=[]
        for t in texts:
            ids=[self._id(x) for x in self._tokens(t)[:self.max_tokens]] or [0]
            seqs.append(ids)
        L=max(len(x) for x in seqs)
        ids=torch.zeros((len(seqs),L),dtype=torch.long,device=device)
        lengths=[]
        for i,s in enumerate(seqs):
            ids[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device); lengths.append(len(s))
        return ids, torch.tensor(lengths,dtype=torch.long,device=device)
    def forward(self,texts,device):
        ids,lengths=self.encode_ids(texts,device)
        x=self.emb(ids)
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed)
        if self.gru.bidirectional:
            z=torch.cat([h[-2],h[-1]],dim=-1)
        else: z=h[-1]
        return self.proj(z)

class MMMILCA(nn.Module):
    """Exam-level MIL student with training-only content-aware report teacher."""
    def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dim=256, heads=4, text_cfg=None):
        super().__init__()
        text_cfg=text_cfg or {}
        self.encoder=ImageEncoder(backbone,pretrained,hidden_dim)
        self.pool=AttentionPool(hidden_dim)
        self.teacher=ClinicalTextEncoder(hidden_dim,
            text_cfg.get('vocab_size',16384),text_cfg.get('embedding_dim',192),
            text_cfg.get('max_tokens',256),text_cfg.get('bidirectional',True))
        self.cross=nn.MultiheadAttention(hidden_dim,heads,batch_first=True)
        self.image_cls=nn.Linear(hidden_dim,NUM_LABELS)
        self.teacher_cls=nn.Linear(hidden_dim,NUM_LABELS)
    def forward(self,images,view_mask,teacher_findings_text=None,training_with_text=False,**kwargs):
        b,n,c,h,w=images.shape
        v=self.encoder(images.reshape(b*n,c,h,w)).reshape(b,n,-1)
        z,a=self.pool(v,view_mask)
        out={"logits":self.image_cls(z),"embedding":z,"view_attention":a}
        if training_with_text and teacher_findings_text is not None:
            t=self.teacher(teacher_findings_text,z.device)
            fused,_=self.cross(t.unsqueeze(1),v,v,key_padding_mask=~view_mask)
            tf=0.5*(fused.squeeze(1)+t)
            out["teacher_logits"]=self.teacher_cls(tf)
            out["teacher_embedding"]=tf
        return out

class HeteroEvidenceDiagnosisGraph(nn.Module):
    """Bipartite heterogeneous message passing over 26 evidence and 7 diagnosis nodes."""
    def __init__(self, image_dim=256, graph_dim=128, steps=2, dropout=0.1):
        super().__init__()
        self.steps=int(steps)
        self.ev_base=nn.Parameter(torch.randn(NUM_EVIDENCE,graph_dim)*0.02)
        self.diag_init=nn.Linear(image_dim,NUM_LABELS*graph_dim)
        self.ev_to_diag_logits=nn.Parameter(torch.zeros(NUM_EVIDENCE,NUM_LABELS))
        self.diag_to_ev_logits=nn.Parameter(torch.zeros(NUM_LABELS,NUM_EVIDENCE))
        nn.init.normal_(self.ev_to_diag_logits,std=.02); nn.init.normal_(self.diag_to_ev_logits,std=.02)
        self.ev_update=nn.GRUCell(graph_dim,graph_dim)
        self.diag_update=nn.GRUCell(graph_dim,graph_dim)
        self.ev_msg=nn.Linear(graph_dim,graph_dim,bias=False)
        self.diag_msg=nn.Linear(graph_dim,graph_dim,bias=False)
        self.drop=nn.Dropout(dropout)
        self.readout=nn.Linear(graph_dim,1)
    def forward(self,z,evidence_probs):
        b=z.size(0); gd=self.ev_base.size(-1)
        ev=evidence_probs.unsqueeze(-1)*self.ev_base.unsqueeze(0)
        dg=self.diag_init(z).view(b,NUM_LABELS,gd)
        a_ed=torch.softmax(self.ev_to_diag_logits,dim=0) # evidence -> each diagnosis
        a_de=torch.softmax(self.diag_to_ev_logits,dim=0) # diagnosis -> each evidence
        for _ in range(self.steps):
            m_d=torch.einsum('bed,el->bld',self.ev_msg(ev),a_ed)
            m_e=torch.einsum('bld,le->bed',self.diag_msg(dg),a_de)
            dg=self.diag_update(self.drop(m_d).reshape(-1,gd),dg.reshape(-1,gd)).view(b,NUM_LABELS,gd)
            ev=self.ev_update(self.drop(m_e).reshape(-1,gd),ev.reshape(-1,gd)).view(b,NUM_EVIDENCE,gd)
        return self.readout(dg).squeeze(-1), dg, ev, a_ed

class RGHGNN(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dim=256, graph_steps=2, graph_hidden_dim=128, graph_dropout=.1):
        super().__init__()
        self.base=BaseMIL(backbone,pretrained,hidden_dim,attention=True)
        self.evidence_head=nn.Linear(hidden_dim,NUM_EVIDENCE)
        self.direct_head=nn.Linear(hidden_dim,NUM_LABELS)
        self.graph=HeteroEvidenceDiagnosisGraph(hidden_dim,graph_hidden_dim,graph_steps,graph_dropout)
        self.gate=nn.Parameter(torch.tensor(0.0))
    def forward(self,images,view_mask,disable_graph=False,**kwargs):
        x=self.base(images,view_mask)
        e_logits=self.evidence_head(x["embedding"]); e=torch.sigmoid(e_logits)
        direct=self.direct_head(x["embedding"])
        graph_logits,dnodes,enodes,adj=self.graph(x["embedding"],e)
        logits=direct if disable_graph else direct+torch.sigmoid(self.gate)*graph_logits
        x.update({"logits":logits,"evidence_logits":e_logits,"direct_logits":direct,
                  "graph_logits":graph_logits,"diagnosis_nodes":dnodes,"evidence_nodes":enodes,
                  "evidence_to_diagnosis_attention":adj})
        return x

class MVConsMIL(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dim=256, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.encoder=ImageEncoder(backbone,pretrained,hidden_dim)
        layer=nn.TransformerEncoderLayer(hidden_dim,heads,hidden_dim*4,dropout,batch_first=True,norm_first=True)
        self.tx=nn.TransformerEncoder(layer,layers)
        self.pool=AttentionPool(hidden_dim); self.cls=nn.Linear(hidden_dim,NUM_LABELS)
    def _aggregate(self,v,mask):
        zv=self.tx(v,src_key_padding_mask=~mask); z,a=self.pool(zv,mask)
        return self.cls(z),z,a,zv
    def forward(self,images,view_mask,view_dropout=0.0,**kwargs):
        b,n,c,h,w=images.shape
        v=self.encoder(images.reshape(b*n,c,h,w)).reshape(b,n,-1)
        logits,z,a,zviews=self._aggregate(v,view_mask)
        out={"logits":logits,"embedding":z,"view_attention":a,"context_views":zviews}
        if self.training and view_dropout>0:
            mask=view_mask.clone(); drop=(torch.rand_like(mask.float())<view_dropout)&mask
            for i in range(b):
                if (mask[i]&~drop[i]).sum()==0:
                    drop[i,torch.where(mask[i])[0][0]]=False
            mask2=mask&~drop
            lg2,z2,a2,_=self._aggregate(v,mask2)
            out.update({"consistency_logits":lg2,"consistency_embedding":z2,"dropped_view_mask":mask2})
        return out

class ProtoRAGNet(nn.Module):
    """Prototype MIL plus checkpoint-persistent training-only case retrieval memory."""
    def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dim=256, prototypes_per_label=4,
                 retrieval_top_k=5, memory_size=4096, retrieval_temperature=.10,
                 prototype_weight=.20, retrieval_weight=.15):
        super().__init__()
        self.base=BaseMIL(backbone,pretrained,hidden_dim,attention=True)
        self.prototypes=nn.Parameter(torch.randn(NUM_LABELS,prototypes_per_label,hidden_dim)*0.02)
        self.direct=nn.Linear(hidden_dim,NUM_LABELS)
        self.proto_scale=nn.Parameter(torch.tensor(10.0))
        self.retrieval_proj=nn.Linear(hidden_dim,NUM_LABELS)
        self.top_k=int(retrieval_top_k); self.memory_size=int(memory_size); self.temperature=float(retrieval_temperature)
        self.prototype_weight=float(prototype_weight); self.retrieval_weight=float(retrieval_weight)
        self.register_buffer('memory_embeddings',torch.zeros(memory_size,hidden_dim),persistent=True)
        self.register_buffer('memory_targets',torch.zeros(memory_size,NUM_LABELS),persistent=True)
        self.register_buffer('memory_masks',torch.zeros(memory_size,NUM_LABELS),persistent=True)
        self.register_buffer('memory_count',torch.zeros((),dtype=torch.long),persistent=True)
        self.register_buffer('memory_ptr',torch.zeros((),dtype=torch.long),persistent=True)
    def _retrieve(self,z):
        n=int(self.memory_count.item())
        if n==0:
            return torch.zeros_like(z), torch.zeros(z.size(0),NUM_LABELS,device=z.device), None
        mem=F.normalize(self.memory_embeddings[:n],dim=-1)
        q=F.normalize(z,dim=-1); sim=q@mem.T
        k=min(self.top_k,n); vals,idx=sim.topk(k,dim=1)
        w=torch.softmax(vals/max(self.temperature,1e-4),dim=1)
        rem=(w.unsqueeze(-1)*self.memory_embeddings[idx]).sum(1)
        mt=self.memory_targets[idx]; mm=self.memory_masks[idx]
        num=(w.unsqueeze(-1)*mt*mm).sum(1); den=(w.unsqueeze(-1)*mm).sum(1).clamp_min(1e-6)
        prior=num/den
        return rem,prior,idx
    @torch.no_grad()
    def _update_memory(self,z,targets,masks):
        z=z.detach(); targets=targets.detach(); masks=masks.detach(); b=z.size(0)
        for i in range(b):
            p=int(self.memory_ptr.item())
            self.memory_embeddings[p].copy_(z[i]); self.memory_targets[p].copy_(targets[i]); self.memory_masks[p].copy_(masks[i])
            self.memory_ptr.fill_((p+1)%self.memory_size)
            self.memory_count.fill_(min(self.memory_size,int(self.memory_count.item())+1))
    def forward(self,images,view_mask,retrieval_targets=None,retrieval_masks=None,retrieval_update=False,disable_retrieval=False,**kwargs):
        x=self.base(images,view_mask); z=x['embedding']
        zn=F.normalize(z,dim=-1); p=F.normalize(self.prototypes,dim=-1)
        sim=torch.einsum('bd,lpd->blp',zn,p).max(-1).values
        proto_logits=self.proto_scale.clamp(1,50)*sim
        rem,prior,idx=self._retrieve(z)
        retrieval_logits=self.retrieval_proj(rem)+torch.logit(prior.clamp(1e-4,1-1e-4))
        if disable_retrieval or int(self.memory_count.item())==0: retrieval_logits=torch.zeros_like(proto_logits)
        logits=self.direct(z)+self.prototype_weight*proto_logits+self.retrieval_weight*retrieval_logits
        x.update({"logits":logits,"prototype_similarity":sim,"prototype_logits":proto_logits,
                  "retrieval_logits":retrieval_logits,"retrieved_indices":idx,
                  "retrieval_memory_count":int(self.memory_count.item())})
        if self.training and retrieval_update and retrieval_targets is not None and retrieval_masks is not None:
            self._update_memory(z,retrieval_targets,retrieval_masks)
        return x

def build_model(name, cfg):
    b=cfg['image']['backbone']; pt=cfg['image'].get('pretrained',True); h=int(cfg.get('hidden_dim',256))
    if name=='baseline_mean_mil': return BaseMIL(b,pt,h,attention=False)
    if name=='baseline_attention_mil': return BaseMIL(b,pt,h,attention=True)
    if name=='mm_mil_ca': return MMMILCA(b,pt,h,int(cfg.get('attention_heads',4)),cfg.get('text_encoder',{}))
    if name=='rg_hgnn': return RGHGNN(b,pt,h,int(cfg.get('graph_steps',2)),int(cfg.get('graph_hidden_dim',128)),float(cfg.get('graph_dropout',.1)))
    if name=='mv_consmil': return MVConsMIL(b,pt,h,int(cfg.get('attention_heads',4)),int(cfg.get('transformer_layers',2)),float(cfg.get('dropout',.2)))
    if name=='protorag_net': return ProtoRAGNet(b,pt,h,int(cfg.get('prototype_count_per_label',4)),int(cfg.get('retrieval_top_k',5)),int(cfg.get('retrieval_memory_size',4096)),float(cfg.get('retrieval_temperature',.10)),float(cfg.get('prototype_weight',.20)),float(cfg.get('retrieval_weight',.15)))
    raise ValueError(name)
