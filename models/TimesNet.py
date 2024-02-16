import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # shape of xf: [B, T, C]
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.k = configs.top_k
        self.k = 3
        self.d_ff = 512
        self.num_kernels = 4
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.hidden_dim, self.d_ff,
                               num_kernels=self.num_kernels),
            nn.GELU(),
            Inception_Block_V1(self.d_ff, configs.hidden_dim,
                               num_kernels=self.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        if self.task_name == 'encoding':
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.hidden_dim, configs.embed, configs.freq,
                                               configs.dropout, configs.condition_dim)
        else:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.hidden_dim, configs.embed, configs.freq,
                                               configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.hidden_dim)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.hidden_dim, configs.c_out, bias=True)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.hidden_dim * configs.seq_len, configs.num_class)
        if self.task_name == 'tick_wise_classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.hidden_dim, configs.num_class)
        if self.task_name == 'encoding':
            self.encode_linear = nn.Linear(
                self.seq_len, self.seq_len)
            self.projection = nn.Linear(
                configs.hidden_dim, configs.hidden_dim, bias=True)
            self.condition_modulation_linear = nn.Linear(
                configs.condition_dim + configs.hidden_dim, configs.hidden_dim, bias=True)
        elif self.task_name == 'encoding_to_hidden':
            self.encode_linear = nn.Linear(
                self.seq_len, self.seq_len + self.pred_len)
            self.projection = nn.Linear(
                configs.hidden_dim, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        # shape of x_enc: [B, T, C]
        means = x_enc.mean(1, keepdim=True).detach()
        # shape of means: [B, 1, C]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        # shape of enc_out: [B, T, C]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        # size of dec_out: [B, T, C]
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # enc_out shape: [B, T, C]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        # output shape: [B, T, C]
        output = self.dropout(output)
        # zero-out padding embeddings

        output = output * x_mark_enc.unsqueeze(-1)
        # output shape: [B, T, C]
        # (batch_size, seq_length * hidden_dim)
        output = output.reshape(output.shape[0], -1)
        # output shape: [B, T*C]
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def tick_wise_classification(self, x_enc, x_mark_enc):
        # embedding
        # print('x_enc:',x_enc.shape,'x_mark_enc:',x_mark_enc.shape)
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # enc_out shape: [B, T, C]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        # output shape: [B, T, C]
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # output shape: [B, T, C]
        output = self.projection(output)
        # output shape: [B, T, N]
        return output

    def encoding_to_hidden(self, x_enc, x_mark_enc, condition_features=None):

        if condition_features is not None:
            enc_out = self.enc_embedding(x_enc, x_mark_enc, condition_features)
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # embedding
        # shape of enc_out: [B, T, C]

        enc_out = self.encode_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)

        return dec_out

    def encoding(self, x_enc, x_mark_enc, condition_features=None):

        if condition_features is not None:
            enc_out = self.enc_embedding(x_enc, x_mark_enc, condition_features)
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # embedding
        # shape of enc_out: [B, T, C]
        enc_out = self.encode_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        if condition_features is not None:
            # modulate condition features with enc_out a
            print('condition modulating')
            enc_out = self.condition_modulation_linear(torch.cat([enc_out, condition_features], dim=-1))
        dec_out = self.projection(enc_out)

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, condition_features=None):

        x_mark_enc = torch.ones_like(x_enc[:, :, 0])
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'tick_wise_classification':
            dec_out = self.tick_wise_classification(x_enc, x_mark_enc)
            return dec_out  # [B, T, N]
        if self.task_name == 'encoding':
            dec_out = self.encoding(x_enc, x_mark_enc, condition_features)
            return dec_out
        if self.task_name == 'encoding_to_hidden':
            dec_out = self.encoding_to_hidden(x_enc, x_mark_enc)
            return dec_out
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
