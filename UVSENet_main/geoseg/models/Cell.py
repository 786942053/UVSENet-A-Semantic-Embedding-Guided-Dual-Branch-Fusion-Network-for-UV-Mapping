import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP_Attention(nn.Module):
    """MLP Attention Module"""

    def __init__(self, channel, dim, mid_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, channel, 1)
        self.mid = mid_dim
        self.linear_0 = nn.Conv1d(channel, self.mid, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.mid, channel, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, bias=False),
            nn.LayerNorm([channel, dim])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)
        x = self.linear_1(attn)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x


def trans(kernel=3, strd=2):
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=kernel, stride=strd, padding=(kernel - 1) // 2, bias=False),
        nn.Softmax(dim=-1)
    )


# class bwd_LSTM(nn.Module):
#     """Backward hierarchical LSTM"""

#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.cell_1 = nn.LSTMCell(input_size=input_dim, hidden_size=output_dim)
#         self.cell_2 = nn.LSTMCell(input_size=input_dim // 2, hidden_size=output_dim // 2)
#         self.cell_3 = nn.LSTMCell(input_size=input_dim // 4, hidden_size=output_dim // 4)
#         self.cell_4 = nn.LSTMCell(input_size=input_dim // 8, hidden_size=output_dim // 8)

#         self.trans2_1h = trans(3, 1)
#         self.trans2_1c = trans(3, 1)
#         self.trans3_2h = trans(3, 1)
#         self.trans3_2c = trans(3, 1)
#         self.trans4_3h = trans(3, 1)
#         self.trans4_3c = trans(3, 1)

#     def forward(self, spec):
#         # Ensure [B, input_dim]
#         if spec.dim() == 1:
#             spec = spec.unsqueeze(0)
#         B = spec.shape[0]

#         # split levels
#         x1 = spec.unsqueeze(-1)
#         x2 = torch.zeros(B, self.input_dim // 2, 2, device=spec.device)
#         x3 = torch.zeros(B, self.input_dim // 4, 4, device=spec.device)
#         x4 = torch.zeros(B, self.input_dim // 8, 8, device=spec.device)

#         # fill x2
#         start, end = 0, self.input_dim // 2
#         for i in range(2):
#             x2[:, :, i] = spec[:, start:end]
#             start = end
#             end += self.input_dim // 2

#         # fill x3
#         start, end = 0, self.input_dim // 4
#         for i in range(4):
#             x3[:, :, i] = spec[:, start:end]
#             start = end
#             end += self.input_dim // 4

#         # fill x4
#         start, end = 0, self.input_dim // 8
#         for i in range(8):
#             x4[:, :, i] = spec[:, start:end]
#             start = end
#             end += self.input_dim // 8

#         hx_backup, cx_backup = [], []
#         out_temp_hx, out_temp_cx = [], []

#         # Level 4
#         out4 = []
#         for i in range(x4.shape[2]):
#             hx4, cx4 = self.cell_4(x4[:, :, i])
#             out_temp_hx.append(hx4)
#             out_temp_cx.append(cx4)
#             if i % 2 == 1:
#                 temp_hx = self.trans4_3h(torch.cat(out_temp_hx, dim=-1).unsqueeze(1)).squeeze()
#                 temp_cx = self.trans4_3c(torch.cat(out_temp_cx, dim=-1).unsqueeze(1)).squeeze()
#                 # Expand to batch size
#                 temp_hx = temp_hx.unsqueeze(0).expand(B, -1)
#                 temp_cx = temp_cx.unsqueeze(0).expand(B, -1)
#                 hx_backup.append(temp_hx)
#                 cx_backup.append(temp_cx)
#                 out_temp_hx.clear()
#                 out_temp_cx.clear()
#             out4.append(hx4)

#         # Level 3
#         out3 = []
#         temp_hx_list, temp_cx_list = [], []
#         for i in range(x3.shape[2]):
#             hx3, cx3 = self.cell_3(x3[:, :, i], (hx_backup[i], cx_backup[i]))
#             temp_hx_list.append(hx3)
#             temp_cx_list.append(cx3)
#             if i % 2 == 1:
#                 temp_hx = self.trans3_2h(torch.cat(temp_hx_list, dim=-1).unsqueeze(1)).squeeze()
#                 temp_cx = self.trans3_2c(torch.cat(temp_cx_list, dim=-1).unsqueeze(1)).squeeze()
#                 temp_hx = temp_hx.unsqueeze(0).expand(B, -1)
#                 temp_cx = temp_cx.unsqueeze(0).expand(B, -1)
#                 hx_backup.append(temp_hx)
#                 cx_backup.append(temp_cx)
#                 temp_hx_list.clear()
#                 temp_cx_list.clear()
#             out3.append(hx3)
#         del hx_backup[:x4.shape[2] // 2]
#         del cx_backup[:x4.shape[2] // 2]

#         # Level 2
#         out2 = []
#         temp_hx_list, temp_cx_list = [], []
#         for i in range(x2.shape[2]):
#             hx2, cx2 = self.cell_2(x2[:, :, i], (hx_backup[i], cx_backup[i]))
#             temp_hx_list.append(hx2)
#             temp_cx_list.append(cx2)
#             if i % 2 == 1:
#                 temp_hx = self.trans2_1h(torch.cat(temp_hx_list, dim=-1).unsqueeze(1)).squeeze()
#                 temp_cx = self.trans2_1c(torch.cat(temp_cx_list, dim=-1).unsqueeze(1)).squeeze()
#                 temp_hx = temp_hx.unsqueeze(0).expand(B, -1)
#                 temp_cx = temp_cx.unsqueeze(0).expand(B, -1)
#                 hx_backup.append(temp_hx)
#                 cx_backup.append(temp_cx)
#                 temp_hx_list.clear()
#                 temp_cx_list.clear()
#             out2.append(hx2)
#         del hx_backup[:x3.shape[2] // 2]
#         del cx_backup[:x3.shape[2] // 2]

#         # Level 1
#         out1 = []
#         for i in range(x1.shape[2]):
#             hx1, cx1 = self.cell_1(x1[:, :, i])
#             out1.append(hx1)

#         return torch.cat(out1, dim=-1)
class bwd_LSTM(nn.Module):
    """Backward hierarchical LSTM"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cell_1 = nn.LSTMCell(input_size=input_dim, hidden_size=output_dim)
        self.cell_2 = nn.LSTMCell(input_size=input_dim // 2, hidden_size=output_dim // 2)
        self.cell_3 = nn.LSTMCell(input_size=input_dim // 4, hidden_size=output_dim // 4)
        self.cell_4 = nn.LSTMCell(input_size=input_dim // 8, hidden_size=output_dim // 8)

        self.trans2_1h = trans(3, 1)
        self.trans2_1c = trans(3, 1)
        self.trans3_2h = trans(3, 1)
        self.trans3_2c = trans(3, 1)
        self.trans4_3h = trans(3, 1)
        self.trans4_3c = trans(3, 1)

    def forward(self, spec):
        if spec.dim() == 1:
            spec = spec.unsqueeze(0)
        B = spec.shape[0]

        x1 = spec.unsqueeze(-1)
        x2 = torch.zeros(B, self.input_dim // 2, 2, device=spec.device)
        x3 = torch.zeros(B, self.input_dim // 4, 4, device=spec.device)
        x4 = torch.zeros(B, self.input_dim // 8, 8, device=spec.device)

        start = 0
        step = self.input_dim // 2
        for i in range(2):
            x2[:, :, i] = spec[:, start:start+step]
            start += step

        start = 0
        step = self.input_dim // 4
        for i in range(4):
            x3[:, :, i] = spec[:, start:start+step]
            start += step

        start = 0
        step = self.input_dim // 8
        for i in range(8):
            x4[:, :, i] = spec[:, start:start+step]
            start += step

        hx_backup, cx_backup = [], []
        out_temp_hx, out_temp_cx = [], []

        # Level 4
        out4 = []
        for i in range(8):
            hx4, cx4 = self.cell_4(x4[:, :, i])
            out_temp_hx.append(hx4)
            out_temp_cx.append(cx4)

            if i % 2 == 1:
                temp_hx = self.trans4_3h(torch.cat(out_temp_hx, dim=-1).unsqueeze(1)).squeeze(1)
                temp_cx = self.trans4_3c(torch.cat(out_temp_cx, dim=-1).unsqueeze(1)).squeeze(1)
                hx_backup.append(temp_hx)
                cx_backup.append(temp_cx)
                out_temp_hx.clear()
                out_temp_cx.clear()

            out4.append(hx4)

        # Level 3
        out3 = []
        temp_h, temp_c = [], []
        for i in range(4):
            hx3, cx3 = self.cell_3(x3[:, :, i], (hx_backup[i], cx_backup[i]))
            temp_h.append(hx3)
            temp_c.append(cx3)

            if i % 2 == 1:
                temp_hx = self.trans3_2h(torch.cat(temp_h, dim=-1).unsqueeze(1)).squeeze(1)
                temp_cx = self.trans3_2c(torch.cat(temp_c, dim=-1).unsqueeze(1)).squeeze(1)
                hx_backup.append(temp_hx)
                cx_backup.append(temp_cx)
                temp_h.clear()
                temp_c.clear()

            out3.append(hx3)

        del hx_backup[:4]
        del cx_backup[:4]

        # Level 2
        out2 = []
        temp_h, temp_c = [], []
        for i in range(2):
            hx2, cx2 = self.cell_2(x2[:, :, i], (hx_backup[i], cx_backup[i]))
            temp_h.append(hx2)
            temp_c.append(cx2)

            if i % 2 == 1:
                temp_hx = self.trans2_1h(torch.cat(temp_h, dim=-1).unsqueeze(1)).squeeze(1)
                temp_cx = self.trans2_1c(torch.cat(temp_c, dim=-1).unsqueeze(1)).squeeze(1)
                hx_backup.append(temp_hx)
                cx_backup.append(temp_cx)
                temp_h.clear()
                temp_c.clear()

            out2.append(hx2)

        del hx_backup[:2]
        del cx_backup[:2]

        # Level 1
        out1 = []
        for i in range(1):
            hx1, cx1 = self.cell_1(x1[:, :, i])
            out1.append(hx1)

        return torch.cat(out1, dim=-1)


class fwd_LSTM(nn.Module):
    """Forward hierarchical LSTM"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cell_1 = nn.LSTMCell(input_size=input_dim, hidden_size=output_dim)
        self.cell_2 = nn.LSTMCell(input_size=input_dim // 2, hidden_size=output_dim // 2)
        self.cell_3 = nn.LSTMCell(input_size=input_dim // 4, hidden_size=output_dim // 4)
        self.cell_4 = nn.LSTMCell(input_size=input_dim // 8, hidden_size=output_dim // 8)

        self.trans1_2h = trans(3, 2)
        self.trans1_2c = trans(3, 2)
        self.trans2_3h = trans(3, 2)
        self.trans2_3c = trans(3, 2)
        self.trans3_4h = trans(3, 2)
        self.trans3_4c = trans(3, 2)

    def forward(self, spec):
        if spec.dim() == 1:
            spec = spec.unsqueeze(0)
        B = spec.shape[0]

        x1 = spec.unsqueeze(-1)
        x2 = torch.zeros(B, self.input_dim // 2, 2, device=spec.device)
        x3 = torch.zeros(B, self.input_dim // 4, 4, device=spec.device)
        x4 = torch.zeros(B, self.input_dim // 8, 8, device=spec.device)

        start, end = 0, self.input_dim // 2
        for i in range(2):
            x2[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 2

        start, end = 0, self.input_dim // 4
        for i in range(4):
            x3[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 4

        start, end = 0, self.input_dim // 8
        for i in range(8):
            x4[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 8

        out1, out2, out3, out4 = [], [], [], []
        hx_backup, cx_backup = [], []

        # Level 1
        for i in range(x1.shape[2]):
            hx1, cx1 = self.cell_1(x1[:, :, i])
            temp_hx = self.trans1_2h(hx1.unsqueeze(1)).squeeze(1)  # shape [B, H]
            temp_cx = self.trans1_2c(cx1.unsqueeze(1)).squeeze(1)  # shape [B, H]
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out1.append(cx1)
        # Level 2
        for i in range(x2.shape[2]):
            index = i // 2
            hx2, cx2 = self.cell_2(x2[:, :, i], (hx_backup[index], cx_backup[index]))
            temp_hx = self.trans2_3h(hx2.unsqueeze(1)).squeeze(1)
            temp_cx = self.trans2_3c(cx2.unsqueeze(1)).squeeze(1)
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out2.append(cx2)
        del hx_backup[:x1.shape[2]]
        del cx_backup[:x1.shape[2]]

        # Level 3
        for i in range(x3.shape[2]):
            index = i // 2
            hx3, cx3 = self.cell_3(x3[:, :, i], (hx_backup[index], cx_backup[index]))
            temp_hx = self.trans3_4h(hx3.unsqueeze(1)).squeeze(1)
            temp_cx = self.trans3_4c(cx3.unsqueeze(1)).squeeze(1)
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out3.append(cx3)
        del hx_backup[:x2.shape[2]]
        del cx_backup[:x2.shape[2]]

        # Level 4
        for i in range(x4.shape[2]):
            index = i // 2
            hx4, cx4 = self.cell_4(x4[:, :, i], (hx_backup[index], cx_backup[index]))
            out4.append(cx4)

        return torch.cat(out4, dim=-1)


class Spectral(nn.Module):
    """Spectral module"""

    def __init__(self, bands, class_count):
        super().__init__()
        self.down_ch = 128
        self.pre = nn.Linear(bands, self.down_ch)
        self.pre_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.recent = fwd_LSTM(self.down_ch, self.down_ch // 2)
        self.longterm = bwd_LSTM(self.down_ch, self.down_ch // 2)
        self.LSTM = nn.LSTM(self.down_ch, self.down_ch // 2, num_layers=1, dropout=0, batch_first=True)
        self.Attention_hx = MLP_Attention(1, self.down_ch // 2, self.down_ch // 4)
        self.Attention_cx = MLP_Attention(1, self.down_ch // 2, self.down_ch // 4)
        self.midhx_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.midcx_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.fc = nn.Linear(self.down_ch // 2, class_count)

    def forward(self, spec):
        # spec: [B, bands]
        x = self.pre(spec)
        x = self.pre_conv(x.unsqueeze(1)).squeeze(1)

        hx = self.recent(x).unsqueeze(1)
        cx = self.longterm(x).unsqueeze(1)

        hx = self.midhx_conv(hx)
        cx = self.midcx_conv(cx)

        hx = self.Attention_hx(hx).permute(1, 0, 2)
        cx = self.Attention_cx(cx).permute(1, 0, 2)

        out, (hx_out, cx_out) = self.LSTM(x.unsqueeze(1), (hx, cx))
        score = self.fc(out.squeeze(1))
        return score
