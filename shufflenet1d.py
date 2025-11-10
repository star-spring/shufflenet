import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batchsize, num_channels, length = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, length)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, length)
    return x


class InvertedResidual1D(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual1D, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        branch_features = oup // 2
        if self.stride == 1:
            assert inp == oup

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv1d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.Conv1d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm1d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv1d(inp if self.stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_features, branch_features, 3, stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.Conv1d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2_1D(nn.Module):
    def __init__(self, num_classes=4, input_channels=1):
        super(ShuffleNetV2_1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 24, 3, 2, 1, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(24, 116, 4)
        self.stage3 = self._make_stage(116, 232, 8)
        self.stage4 = self._make_stage(232, 464, 4)

        self.conv5 = nn.Sequential(
            nn.Conv1d(464, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, inp, oup, repeats):
        layers = [InvertedResidual1D(inp, oup, 2)]
        for i in range(repeats - 1):
            layers.append(InvertedResidual1D(oup, oup, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
