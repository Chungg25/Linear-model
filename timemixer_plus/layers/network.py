import torch
from torch import nn

class MixerBlock(nn.Module):
    def __init__(self, channel, seq_len, d_model, dropout=0.1, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, seq_len)
        )

    def forward(self, x):
        # x: [B, C, seq_len]
        x_norm = self.norm(x) 
        z = self.mlp(x_norm)  
        out = x + z           
        return out

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.activation = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
        )

        self.mixer = MixerBlock(channel=self.enc_in, seq_len=self.seq_len, d_model=self.d_model, dropout=dropout)


        # Chỉ giữ lại các khối embedding, bỏ MLP và fc8
        # self.mlp và self.fc8 không còn được sử dụng
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t_flat = torch.reshape(t, (B*C, I))

        # Seasonal embedding (bỏ MLP)
        s_conv = self.conv1d(s)
        s_pool = self.pool(s_conv)
        s_act = self.activation(s_pool)
        s_emb = self.mixer(s_act + s)

        # Trend embedding (bỏ fc8)
        t_emb = self.fc5(t_flat)
        t_emb = self.gelu1(t_emb)
        t_emb = self.ln1(t_emb)
        t_emb = self.fc7(t_emb)
        t_emb = torch.reshape(t_emb, (B, C, self.pred_len))
        t_emb = t_emb.permute(0,2,1)

        # Trả về 2 embedding: seasonal và trend
        return s_emb, t_emb