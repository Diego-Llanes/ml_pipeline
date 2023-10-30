"""
main class for building a DL pipeline.

"""

from torch.utils.data import DataLoader
from torch.optim import AdamW
from data import GenericDataset
from model.linear import ClassificationDNN
from model.cnn import ClassificationCNN
from runner import Runner
import wandb
import argparse
import matplotlib.pyplot as plt
import random
import torch

from model.linear import activations

# Set the global seed
torch.manual_seed(0)


def train(args):

    if not args.debug:
        wandb.init(
            project=args.exp,
            name=args.run,
            config=args,
            entity=args.entity
        )

    train_set = GenericDataset('train')
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)

    dev_set = GenericDataset('dev')
    dev_loader = DataLoader(dev_set, batch_size=args.bs, shuffle=True)

    in_features, out_features = train_set.get_in_out_size()

    model = ClassificationCNN(out_features, activation=args.act)
    # model = ClassificationDNN(in_features, args.hsize, out_features, args.nlayers, activation=args.act)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.log_weights and not args.debug:
        wandb.watch(model, log='all')

    train_runner = Runner(
        split='train',
        train_set=train_set,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
    )

    dev_runner = Runner(
        split='dev',
        train_set=dev_set,
        train_loader=dev_loader,
        model=model,
        optimizer=optimizer,
    )

    for _ in range(args.epochs):
        train_stats = train_runner.next()
        dev_stats = dev_runner.next()
        print(f"train_stats: {train_stats}")
        print(f"dev_stats: {dev_stats}\n")
        idx = random.randint(0, len(dev_set))
        random_xy = dev_set[idx]
        pred = model(torch.tensor(random_xy[0]))
        true_label = random_xy[1]
        two_array = dev_set[idx][0].reshape(28, 28)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(two_array, cmap='gray', interpolation='nearest')
        ax.axis('off')
        plt.tight_layout()

        image_path = args.save_dir + "/temp_image.png"
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        predicted_class = pred.argmax().item()
        confidence = 100 * pred.max().item()
        true_label = true_label.argmax().item()

        if not args.debug:
            wandb.log(
                {
                    'toss': train_stats,
                    'voss': dev_stats,
                    'Image': wandb.Image(
                        image_path,
                        caption=f"Prediction: {predicted_class}, Confidence: {confidence:.2f}%, True Label: {true_label}"),
                }
            )

    torch.save(model.state_dict(), args.save_dir + '/model_state_dict.pth')

    if not args.debug:
        wandb.save(args.save_dir + '/model_state_dict.pth')
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--hsize", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--act", type=str, default="ReLU", choices=list(activations.keys()))
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=10)

    # WandB Flags (advanced) 😎
    parser.add_argument('--entity', default='diegollanes', help='wandb entity')
    parser.add_argument('--save_dir', default='save', help='where to save stuff')
    parser.add_argument('--log_weights', action='store_true', help='help me')
    parser.add_argument("--exp", type=str, default="mnist",
                        help="Will group all run under this experiment name.")
    parser.add_argument("--run", type=str, default=None,
                        help="Some name to tell what the experiment run was about.")
    parser.add_argument('--debug', action='store_true', help='help me')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
