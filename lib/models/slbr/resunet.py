import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.slbr.blocks import UpConv, DownConv, MBEBlock, SMRBlock, ECABlock

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


class CoarseEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=3, blocks=1, start_filters=32, norm=nn.BatchNorm2d):
        super(CoarseEncoder, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters * (2 ** i)
            # pooling = True if i < depth-1 else False
            pooling = True
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, norm=norm)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs


class SharedBottleNeck(nn.Module):
    def __init__(self, in_channels=512, depth=5, shared_depth=2, blocks=1, residual=True,
                 concat=True, norm=nn.InstanceNorm2d, act=F.relu):
        super(SharedBottleNeck, self).__init__()
        self.down_convs = []
        self.up_convs = []
        self.down_im_atts = []
        self.down_mask_atts = []
        self.up_im_atts = []
        self.up_mask_atts = []

        dilations = [1, 2, 5]
        start_depth = depth - shared_depth
        max_filters = 512
        for i in range(start_depth, depth):  # depth = 5 [0,1,2,3]
            ins = in_channels if i == start_depth else outs
            outs = min(ins * 2, max_filters)
            # Encoder convs
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, norm=norm, dilations=dilations)
            self.down_convs.append(down_conv)

            # Decoder convs
            if i < depth - 1:
                up_conv = UpConv(min(outs * 2, max_filters), outs, blocks, norm=norm, dilations=dilations)
                self.up_convs.append(up_conv)
                self.up_im_atts.append(ECABlock(outs))
                self.up_mask_atts.append(ECABlock(outs))

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # task-specific channel attention blocks
        self.up_im_atts = nn.ModuleList(self.up_im_atts)
        self.up_mask_atts = nn.ModuleList(self.up_mask_atts)

        reset_params(self)

    def forward(self, input):
        # Encoder convs
        im_encoder_outs = []
        mask_encoder_outs = []
        x = input
        for i, d_conv in enumerate(self.down_convs):
            # d_conv, attn = nets
            x, before_pool = d_conv(x)
            im_encoder_outs.append(before_pool)
            mask_encoder_outs.append(before_pool)
        x_im = x
        x_mask = x

        # Decoder convs
        x = x_im
        for i, nets in enumerate(zip(self.up_convs, self.up_im_atts)):
            up_conv, attn = nets
            before_pool = None
            if im_encoder_outs is not None:
                before_pool = im_encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=attn)
        x_im = x

        x = x_mask
        for i, nets in enumerate(zip(self.up_convs, self.up_mask_atts)):
            up_conv, attn = nets
            before_pool = None
            if mask_encoder_outs is not None:
                before_pool = mask_encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=attn)
        x_mask = x

        return x_im, x_mask



class CoarseDecoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1):
        super(CoarseDecoder, self).__init__()
        self.up_convs_bg = nn.ModuleList()
        self.up_convs_mask = nn.ModuleList()
        self.atts_bg = nn.ModuleList()
        self.atts_mask = nn.ModuleList()

        outs = in_channels
        for i in range(depth):
            ins = outs
            outs = ins // 2

            # Background reconstruction branch
            up_conv = MBEBlock(ins, outs, blocks=blocks)
            self.up_convs_bg.append(up_conv)
            self.atts_bg.append(ECABlock(outs))  # Always use attention

            # Mask prediction branch
            up_conv = SMRBlock(ins, outs, blocks=blocks)
            self.up_convs_mask.append(up_conv)
            self.atts_mask.append(ECABlock(outs))  # Always use attention

        # Final convolution
        self.conv_final_bg = nn.Conv2d(outs, out_channels, 1, 1, 0)

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, bg, fg, mask, encoder_outs=None):
        bg_x = bg
        mask_x = mask
        mask_outs = []
        bg_outs = []

        for i, (up_bg, up_mask) in enumerate(zip(self.up_convs_bg, self.up_convs_mask)):
            # Get corresponding encoder features
            before_pool = encoder_outs[-(i + 1)] if encoder_outs is not None else None

            # Process mask branch with attention
            mask_before_pool = self.atts_mask[i](before_pool)
            smr_outs = up_mask(mask_x, mask_before_pool)
            mask_x = smr_outs['feats'][0]
            primary_map, self_calibrated_map = smr_outs['attn_maps']
            mask_outs.extend([primary_map, self_calibrated_map])

            # Process background branch with attention
            bg_before_pool = self.atts_bg[i](before_pool)
            bg_x = up_bg(bg_x, bg_before_pool, self_calibrated_map.detach())
            bg_outs.append(bg_x)

        # Final processing
        bg_x = self.conv_final_bg(bg_x)
        mask_x = mask_outs[-1]

        return [bg_x] + bg_outs, [mask_x] + mask_outs, None


class SLBR(nn.Module):

    def __init__(self, in_channels=3, depth=5, shared_depth=2, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True,
                 concat=True, long_skip=False):
        super(SLBR, self).__init__()
        self.shared = shared_depth = 2
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)

        # coarse stage
        self.encoder = CoarseEncoder(in_channels=in_channels, depth=depth - shared_depth, blocks=blocks[0],
                                     start_filters=start_filters)
        self.shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=blocks[4],
                                               residual=residual,
                                               concat=concat, )

        self.coarse_decoder = CoarseDecoder(in_channels=start_filters * 2 ** (depth - shared_depth),
                                            out_channels=out_channels_image, depth=depth - shared_depth,
                                            blocks=blocks[1])

        self.long_skip = long_skip

    def forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        unshared_before_pool = before_pool  # [: - self.shared]

        im, mask = self.shared_decoder(image_code)
        im_arr, mask_arr, _ = self.coarse_decoder(im, None, mask, unshared_before_pool)
        reconstructed_image = torch.tanh(im_arr[0])
        if self.long_skip:
            reconstructed_image = (reconstructed_image + synthesized).clamp(0, 1)


        return reconstructed_image, mask_arr[0]
