import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict

DEVICE = torch.device("cpu")

class DecoderManual(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU, arch='vit-base'):
        super(DecoderManual, self).__init__()
        if i_dim: self.shared_feature = 1
        else:     self.shared_feature = 0
        if self.shared_feature:
            #! start from 7*7*16(784:16) or 7*7*32(1568:800) or 7*7*64(3,136:2368)
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1) # can be tuned
            src_c = src_dim // 49
        else:
            src_c = src_dim
        
        bias_flag = False
        body_seq = []
        
        if arch in ['vit-mae-base', 'vit-base']:
            if src_c >= 64:    g_c = 64
            else:              g_c = src_c
            body_seq = [  nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                                        nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag),  nn.BatchNorm2d(64), act(),
                                        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias_flag),nn.BatchNorm2d(64), act(),
                                        nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                        nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag),  nn.BatchNorm2d(32), act(),
                                        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=bias_flag),nn.BatchNorm2d(32), act(),
                                        nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                        nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag),  nn.BatchNorm2d(32), act(),
                                        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=bias_flag),nn.BatchNorm2d(32), act(),
                                        nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                        nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag),  nn.BatchNorm2d(16), act(),
                                        nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=bias_flag),nn.BatchNorm2d(16), act(),
                                        nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
            names_list = ['conv_transpose_1',
                              'conv_transpose_2', 'bn_1', 'act_1', 
                              'conv_1', 'bn_2', 'act_2',
                              'conv_transpose_3',
                              'conv_transpose_4', 'bn_3', 'act_3',
                              'conv_2', 'bn_4', 'act_4',
                              'conv_transpose_5',
                              'conv_transpose_6', 'bn_5', 'act_5',
                              'conv_3', 'bn_6', 'act_6',
                              'conv_transpose_7',
                              'conv_transpose_8', 'bn_7', 'act_7',
                              'conv_4', 'bn_8', 'act_8',
                              'conv_transpose_9']
            
        elif arch == 'dino-resnet-50':
            body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(64), act()]
            body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                       nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]            
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(16), act()]
            body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        else: raise ValueError('not implemented')
        assert len(names_list) == len(body_seq), f'Names list and body sequence length mismatched: {len(names_list)} vs {len(body_seq)}'
        dict_list = OrderedDict({names_list[i]: body_seq[i] for i in range(len(names_list))})
        self.body   = nn.Sequential(dict_list)        


    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger.to(DEVICE)), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
        else:
            return self.body(z)
        return self.body(z_cube)


if __name__ == "__main__":
    METHOD = 'coordinator'
    PT_BACKBONE = 'vit-mae-base' # vit-base / vit-mae-base
    SRC_DIM = 1568 # 784 / 1568 / 3136 #? => only for pre-trained Enc
    E_OUT_DIM = 0 # 64 / 128 / 256 #? => only for scratch Enc
    SPSA_PARAMS = [0.0,0.001,40.0,0.6,0.1]
    OPT_TYPE = "spsa-gc" # [spsa, spsa-gc, naive]
    MOMS = 0.9 # first moment scale.
    SP_AVG = 5 # grad estimates averaging steps
    P_EPS = 1.0 # prompt scale

    dm = DecoderManual(768, SRC_DIM, act=nn.GELU, arch=PT_BACKBONE)
    print(summary(dm, (768,), batch_size=1, device='cpu'))
