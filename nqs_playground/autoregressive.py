import os
import torch
import torch.nn.functional as F

from . import core


class CausalConv2d(torch.nn.Conv2d):
    def __init__(self, mask_center: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (_, _, h, w) = self.weight.shape
        mask = torch.zeros_like(self.weight)
        mask.data[:, :, : h // 2, :] = 1
        mask.data[:, :, h // 2, : w // 2 + int(not mask_center)] = 1
        self.register_buffer("mask", mask)
        # Correction to Xavier initialization
        # self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class CausalResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        intermediate_channels = in_channels // 2
        self.block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
            ),
            torch.nn.ReLU(),
            CausalConv2d(
                mask_center=False,
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class PixelCNN(torch.nn.Module):
    def __init__(
        self, input_shape, residual_channels=64, number_residual=1, epsilon=1e-5
    ):
        super().__init__()
        self.input_shape = input_shape
        self.residual_channels = residual_channels
        self.epsilon = epsilon
        self._first = CausalConv2d(
            mask_center=True,
            in_channels=1,
            out_channels=residual_channels,
            kernel_size=7,
            padding=3,
        )
        self._residual_blocks = torch.nn.ModuleList(
            [
                CausalResidualBlock(residual_channels, residual_channels)
                for _ in range(number_residual)
            ]
        )
        self._last = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=residual_channels, out_channels=1, kernel_size=1
            ),
            # torch.nn.Sigmoid(),  # Do we need it?
        )

    def forward(self, x):
        assert x.size()[1:] == self.input_shape
        x = self._first(x)
        for layer in self._residual_blocks:
            x = x + layer(x)
        x = self._last(x)
        return x

    def log_prob(self, x, ŷ):
        batch_size = x.size(0)
        x, ŷ = x.view(batch_size, -1), ŷ.view(batch_size, -1)
        assert not torch.any(torch.isnan(ŷ))
        # ŷ = torch.sigmoid(ŷ)
        # log_prob = (torch.log(ŷ + self.epsilon) * x +
        #             torch.log(1 - ŷ + self.epsilon) * (1 - x))
        # log_prob = log_prob.view(batch_size, -1).sum(dim=1, keepdim=True)
        # assert not torch.any(torch.isnan(log_prob))
        # return log_prob
        loss = F.binary_cross_entropy_with_logits(ŷ, x, reduction="none").sum(
            dim=1, keepdim=True
        )
        return -loss

    @torch.no_grad()
    def sample(self, number_samples: int):
        shape = (number_samples,) + self.input_shape
        device = self._first.weight.device
        conditioned_on = torch.full(shape, -1.0, device=device)

        (_, _, height, width) = shape
        for r in range(height):
            for c in range(width):
                out = self.forward(conditioned_on)[:, :, r, c]
                conditioned_on[:, :, r, c] = torch.where(
                    conditioned_on[:, :, r, c] < 0,
                    torch.distributions.Bernoulli(logits=out).sample(),
                    conditioned_on[:, :, r, c],
                )
        return conditioned_on


class WrapperForQuantum(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._number_spins = self.model.input_shape[-2] * self.model.input_shape[-1]

    def _unpack_and_reshape(self, spin: torch.Tensor) -> torch.Tensor:
        # Unpack ls_bits512 into float32
        spin = torch.ops.tcm.unpack(spin, self._number_spins)
        assert torch.all((spin == 1.0) | (spin == -1.0))
        # Convert +1/-1 to 1/0
        spin = (spin + 1) / 2
        # Reshape
        return spin.view(spin.size(0), *self.model.input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unpack_and_reshape(x)
        assert torch.all((x == 0.0) | (x == 1.0))
        ŷ = self.model.forward(x)
        assert not torch.any(torch.isnan(ŷ))
        return ŷ

    def log_prob(self, x, ŷ=None):
        x = self._unpack_and_reshape(x)
        assert torch.all((x == 0.0) | (x == 1.0))
        if ŷ is None:
            ŷ = self.model.forward(x)
            assert not torch.any(torch.isnan(ŷ))
        return self.model.log_prob(x, ŷ)

    @torch.no_grad()
    def sample(self, number_samples: int, pack: bool = False) -> torch.Tensor:
        sample = self.model.sample(number_samples)
        # Transform 1/0 to +1/-1
        sample = 2 * sample - 1
        # Optionally convert from float32 to ls_bits512
        if pack:
            sample = core.pack(sample.view(sample.size(0), -1))
        else:
            sample = sample.view(number_samples, -1)
        return sample


def get_mnist_loaders(batch_size, dynamically_binarize=False, data_folder="data"):
    import torchvision

    transform = [torchvision.transforms.ToTensor()]
    if dynamically_binarize:
        transform.append(lambda x: torch.distributions.Bernoulli(probs=x).sample())
    transform = torchvision.transforms.Compose(transform)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            data_folder, train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            data_folder, train=False, download=True, transform=transform
        ),
        batch_size=batch_size,
        num_workers=0,
    )
    return train_loader, test_loader


def make_checkpoint(epoch, model, optimizer, log_dir):
    folder = os.path.join(log_dir, "checkpoints")
    filename = "state_{:04d}.pt".format(epoch)
    os.makedirs(folder, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(folder, filename),
    )


def try_sampling(log_dir="run_mnist", device="cuda"):
    import torchvision

    model = PixelCNN(input_shape=(1, 28, 28), number_residual=15, residual_channels=32)
    model.to(device)

    checkpoint = torch.load(
        os.path.join(log_dir, "checkpoints", "state_{:04d}.pt".format(24))
    )
    model.load_state_dict(checkpoint["model"])

    sample = model.sample(144)
    torchvision.utils.save_image(
        sample, "sample_{:02d}.png".format(checkpoint["epoch"]), nrow=12, padding=0
    )


def test_pixelcnn_on_mnist(
    epochs=300, batch_size=256, log_dir="run_mnist", device="cuda",
):
    import torchvision

    train_loader, test_loader = get_mnist_loaders(batch_size, dynamically_binarize=True)
    model = PixelCNN(input_shape=(1, 28, 28), number_residual=15, residual_channels=32)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 0.999977
    )

    def loss_fn(x, y, ŷ):
        r"""
        Parameters:
          x : input
          y : expected output
          ŷ : model output
        """
        batch_size = x.shape[0]
        x, ŷ = x.view(batch_size, -1), ŷ.view(batch_size, -1)
        loss = F.binary_cross_entropy_with_logits(ŷ, x, reduction="none")
        return loss.sum(dim=1).mean()

    for epoch_index in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            ŷ = model(x)
            loss = loss_fn(x, y, ŷ)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            scheduler.step()
        train_loss = total_loss / total_samples
        make_checkpoint(epoch_index, model, optimizer, log_dir)

        model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x = x.to(device)
                y = y.to(device)
                ŷ = model(x)
                loss = loss_fn(x, y, ŷ)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        eval_loss = total_loss / total_samples
        print(
            "Epoch {}: train loss: {}, eval loss: {}...".format(
                epoch_index, train_loss, eval_loss
            )
        )

        sample = model.sample(144)
        torchvision.utils.save_image(
            sample,
            os.path.join(log_dir, "sample_{:02d}.png".format(epoch_index)),
            nrow=12,
            padding=0,
        )


def test_pixelcnn_supervised_4x4(
    epochs=10000, batch_size=32, log_dir="run_4x4", device="cuda",
):
    import lattice_symmetries as ls
    import numpy as np

    basis = ls.SpinBasis(ls.Group([]), number_spins=16, hamming_weight=None)
    # Heisenberg Hamiltonian with flipped sign to avoid sign problem
    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    sites = np.arange(16)
    x = sites % 4
    y = sites // 4
    edges = list(zip(sites, (x + 1) % 4 + 4 * y)) + list(
        zip(sites, x + 4 * ((y + 1) % 4))
    )
    operator = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    print(edges)

    energy, ground_state = ls.diagonalize(operator)
    ground_state = np.abs(ground_state)
    ground_state[ground_state < 10e-9] = 1e-9
    ground_state = ground_state.astype(np.float32)
    ground_state /= np.linalg.norm(ground_state)
    assert not np.any(np.isnan(ground_state))
    target_log_prob = torch.from_numpy(ground_state).log()
    target_log_prob *= 2
    if target_log_prob.dim() == 1:
        target_log_prob.unsqueeze_(dim=1)
    assert not torch.any(torch.isnan(target_log_prob))
    assert not torch.any(torch.isinf(target_log_prob))

    model = WrapperForQuantum(
        PixelCNN(input_shape=(1, 4, 4), number_residual=7, residual_channels=32)
    )
    model.to(device)
    print("Model has {} parameters".format(sum(t.numel() for t in model.state_dict().values())))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    #     optimizer, lr_lambda=lambda _: 0.999977
    # )

    def loss_fn(x, y, ŷ):
        r"""
        Parameters:
          x : input
          y : target log prob
          ŷ : model output
        """
        batch_size = x.size(0)
        ψ = torch.exp(0.5 * model.log_prob(x, ŷ))
        φ = y.view(batch_size, 1)
        return F.mse_loss(ψ, φ)
        # log_prob = model.log_prob(x, ŷ)
        # y = y.view(batch_size, 1)
        # return F.mse_loss(log_prob, y)

    for epoch_index in range(epochs):
        x = model.sample(batch_size, pack=True)
        indices = ls.batched_index(basis, x.cpu().numpy().view(np.uint64))
        indices = torch.from_numpy(indices.view(np.int64))
        y = torch.from_numpy(ground_state[indices]).to(device) # target_log_prob[indices].to(device)

        optimizer.zero_grad()
        ŷ = model(x)
        assert not torch.any(torch.isnan(ŷ))
        loss = loss_fn(x, y, ŷ)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if epoch_index % 100 == 0:
            x = torch.from_numpy(basis.states.view(np.int64)).to(device)
            ŷ = model(x)
            assert not torch.any(torch.isnan(ŷ))
            ψ = torch.exp(0.5 * model.log_prob(x, ŷ)).squeeze()
            assert not torch.any(torch.isnan(ψ))
            φ = torch.from_numpy(ground_state).squeeze().to(device)
            assert torch.isclose(torch.linalg.norm(φ), torch.scalar_tensor(1.0))
            assert not torch.any(torch.isnan(φ))
            print(
                "Epoch {:04d}: norm: {:02f}, loss: {:04f}, overlap: {:04f}".format(
                    epoch_index,
                    torch.linalg.norm(ψ),
                    F.mse_loss(model.log_prob(x, ŷ), target_log_prob.to(device).view(-1, 1)),
                    torch.dot(ψ, φ),
                )
            )
