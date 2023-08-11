import torch
from torchvision.models import resnet18
from torch import nn


class ResNet18(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        # block: Type[Union[BasicBlock, Bottleneck]],
        # layers: List[int],
        num_classes: int = 1000,
        # zero_init_residual: bool = False,
        # groups: int = 1,
        # width_per_group: int = 64,
        # replace_stride_with_dilation: Optional[List[bool]] = None,
        # norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """Initialize a `ResNet18` module.
        """
        super().__init__()

        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        return self.model(x).squeeze()


if __name__ == "__main__":
    _ = ResNet18()
