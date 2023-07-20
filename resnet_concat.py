import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (
                    batch_size,
                    channels,
                    height - (self.block_size - 1),
                    width - (self.block_size - 1),
                )
            )
            if torch.cuda.is_available():
                mask = mask.cuda()
            block_mask = self._compute_block_mask(mask)
            countM = (
                block_mask.size()[0]
                * block_mask.size()[1]
                * block_mask.size()[2]
                * block_mask.size()[3]
            )
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size)
                .view(-1, 1)
                .expand(self.block_size, self.block_size)
                .reshape(-1),  # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), 1)
        if torch.cuda.is_available():
            offsets = offsets.cuda()

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding),
            )
            padded_mask[
                block_idxs[:, 0],
                block_idxs[:, 1],
                block_idxs[:, 2],
                block_idxs[:, 3],
            ] = 1.0
        else:
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding),
            )

        block_mask = 1 - padded_mask
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_pool = use_pool

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.use_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked),
                    1.0 - self.drop_rate,
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size ** 2
                    * feat_size ** 2
                    / (feat_size - self.block_size + 1) ** 2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    
    
    
    
class bottleblock(nn.Module):

    def __init__(
        self,
        inplanes,
        planes=640,
        gmp=True,
    ):
        super(bottleblock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        
        self.gmp = gmp

        self.maxpool = nn.AdaptiveMaxPool2d((5, 5))
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

    def forward(self, x):

        
        if self.gmp:
            out = self.maxpool(x)
        else:
            out = self.avgpool(x)
            
            
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out
    
    
    
class finalblock(nn.Module):
    def __init__(
        self,
        inplanes=640*4,
        planes=640,
    ):
        super(finalblock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        

    def forward(self, x):
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        bottle=bottleblock,
        final=finalblock,
        keep_prob=1.0,
        avg_pool=True,
        drop_rate=0.1,
        dropblock_size=5,
        is_flatten=True,
        maxpool_last2=True,
    ):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
            use_pool=maxpool_last2,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
            use_pool=maxpool_last2,
        )
        
        
        self.bottlev1=bottle(inplanes=64)
        self.bottlev2=bottle(inplanes=160)
        self.bottlev3=bottle(inplanes=320)
        self.bottlev4=final()
        
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.is_flatten = is_flatten
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        stride=1,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                drop_rate,
                drop_block,
                block_size,
                use_pool=use_pool,
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x1 = self.bottlev1(x1)
        x2 = self.bottlev2(x2)
        x3 = self.bottlev3(x3)
        
        x4 = torch.cat([x1,x2,x3,x4], dim=1)
        x4 = self.bottlev4(x4)
        
        if self.keep_avg_pool:
            x4 = self.avgpool(x4)
        if self.is_flatten:
            x4 = x4.view(x.size(0), -1)
        return x4


def resnet12_cat(keep_prob=1.0, avg_pool=True, is_flatten=True, maxpool_last2=True, **kwargs):
# construction of resnet
    model = ResNet(
        BasicBlock,
        bottleblock,
        finalblock,
        keep_prob=keep_prob,
        avg_pool=avg_pool,
        is_flatten=is_flatten,
        maxpool_last2=maxpool_last2,
        **kwargs
    )
    return model


if __name__ == "__main__":
    model = resnet12(avg_pool=True).cuda()
    data = torch.rand(10, 3, 84, 84).cuda()
    output = model(data)
    print(output.size())
