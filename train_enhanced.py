"""
Enhanced Training Script with Topological Analysis
Optimizes HRNet using bottleneck distance monitoring
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add hrnet_base to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hrnet_base', 'lib'))

from topology_analyzer import TopologicalAnalyzer, TopologyAwareTraining
from models.cls_hrnet import HighResolutionNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HRNetCIFAR(nn.Module):
    """Adapted HRNet for CIFAR-10/100 (32x32 images)"""

    def __init__(self, num_classes=10, width=18):
        super(HRNetCIFAR, self).__init__()

        # Configuration for CIFAR
        self.width = width
        cfg = self._get_cifar_config(width, num_classes)

        # Build base HRNet architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        # Simplified HRNet stages for CIFAR
        self.layer1 = self._make_layer(64, 64, 4)

        # Multi-resolution branches
        channels = [width, width*2, width*4, width*8]
        self.transition1 = self._make_transition([64], [channels[0], channels[1]])

        self.stage2 = nn.Sequential(
            self._make_branch_layer(channels[:2], [4, 4]),
        )

        self.transition2 = self._make_transition(channels[:2], channels[:3])

        self.stage3 = nn.Sequential(
            self._make_branch_layer(channels[:3], [4, 4, 4]),
        )

        self.transition3 = self._make_transition(channels[:3], channels[:4])

        self.stage4 = nn.Sequential(
            self._make_branch_layer(channels[:4], [4, 4, 4, 4]),
        )

        # Classification head
        self.incre_modules = nn.ModuleList([
            self._make_layer(channels[i], 128*(2**i), 1) for i in range(4)
        ])

        self.downsamp_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128*(2**i), 128*(2**(i+1)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128*(2**(i+1))),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        self.final_layer = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(2048, num_classes)

        # Initialize weights
        self._init_weights()

    def _get_cifar_config(self, width, num_classes):
        """Generate config for CIFAR"""
        return {
            'width': width,
            'num_classes': num_classes
        }

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        """Create residual layer"""
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def _make_transition(self, num_channels_pre, num_channels_cur):
        """Create transition layer between stages"""
        num_branches_cur = len(num_channels_cur)
        num_branches_pre = len(num_channels_pre)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre[-1]
                    outchannels = num_channels_cur[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_branch_layer(self, num_channels, num_blocks):
        """Create multi-resolution branch"""
        branches = []
        for i, (channels, blocks) in enumerate(zip(num_channels, num_blocks)):
            branches.append(self._make_layer(channels, channels, blocks))
        return nn.ModuleList(branches)

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        """Forward pass with optional feature extraction"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Stage 2
        x_list = []
        for i, transition in enumerate(self.transition1):
            if transition is not None:
                x_list.append(transition(x))
            else:
                x_list.append(x)

        # Process branches correctly
        x_list_processed = []
        for i, (branch, feat) in enumerate(zip(self.stage2[0], x_list)):
            x_list_processed.append(branch(feat))
        x_list = x_list_processed

        # Stage 3
        x_list_new = []
        for i, transition in enumerate(self.transition2):
            if transition is not None:
                if i < len(x_list):
                    x_list_new.append(transition(x_list[i]))
                else:
                    x_list_new.append(transition(x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = x_list_new

        x_list_processed = []
        for i, (branch, feat) in enumerate(zip(self.stage3[0], x_list)):
            x_list_processed.append(branch(feat))
        x_list = x_list_processed

        # Stage 4
        x_list_new = []
        for i, transition in enumerate(self.transition3):
            if transition is not None:
                if i < len(x_list):
                    x_list_new.append(transition(x_list[i]))
                else:
                    x_list_new.append(transition(x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = x_list_new

        x_list_processed = []
        for i, (branch, feat) in enumerate(zip(self.stage4[0], x_list)):
            x_list_processed.append(branch(feat))
        x_list = x_list_processed

        # Classification head
        y = self.incre_modules[0](x_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](x_list[i+1]) + self.downsamp_modules[i](y)

        # Extract features before classification
        features = self.final_layer(y)

        # Global average pooling
        y = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        y = y.view(y.size(0), -1)

        # Classification
        output = self.classifier(y)

        if return_features:
            return output, y  # Return logits and flattened features
        return output


class BasicBlock(nn.Module):
    """Basic residual block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def train_epoch(model, train_loader, criterion, optimizer, epoch, device,
                topology_trainer=None, writer=None, log_interval=100):
    """Train for one epoch with topological monitoring"""
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    running_topo_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with feature extraction
        if topology_trainer is not None:
            output, features = model(data, return_features=True)
            loss, loss_stats = topology_trainer.compute_combined_loss(
                output, target, features, criterion
            )
            topo_loss = loss_stats['topo_loss']
        else:
            output = model(data)
            loss = criterion(output, target)
            topo_loss = 0.0

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / len(data)

        # Update statistics
        running_loss += loss.item()
        running_acc += acc
        running_topo_loss += topo_loss

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2f}%',
            'topo': f'{topo_loss:.4f}'
        })

        # Log to tensorboard
        if writer and batch_idx % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/batch_acc', acc, global_step)
            if topology_trainer:
                writer.add_scalar('train/topo_loss', topo_loss, global_step)

    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    epoch_topo_loss = running_topo_loss / len(train_loader)

    return epoch_loss, epoch_acc, epoch_topo_loss


def validate(model, val_loader, criterion, device, topology_analyzer=None):
    """Validate model with optional topological analysis"""
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    all_features = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)

            # Forward pass
            if topology_analyzer is not None:
                output, features = model(data, return_features=True)
                all_features.append(features.cpu().numpy())
            else:
                output = model(data)

            loss = criterion(output, target)

            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

            all_targets.extend(target.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    # Topological analysis
    topo_stats = {}
    if topology_analyzer is not None and all_features:
        all_features = np.concatenate(all_features, axis=0)
        logger.info("Computing topological features of validation set...")
        topo_stats = topology_analyzer.compute_persistence_diagram(all_features, label='validation')
        if topo_stats:
            logger.info(f"Validation Betti numbers: {topo_stats['betti_numbers']}")
            logger.info(f"Persistence entropy: {topo_stats['persistence_entropy']:.4f}")

    return val_loss, val_acc, topo_stats


def main():
    parser = argparse.ArgumentParser(description='HRNet CIFAR Training with Topological Analysis')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--width', type=int, default=18, help='HRNet width multiplier')
    parser.add_argument('--topology-weight', type=float, default=0.01,
                        help='Weight for topological regularization')
    parser.add_argument('--topology-interval', type=int, default=10,
                        help='Interval for topological analysis (epochs)')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Data preparation
    logger.info(f'Preparing {args.dataset.upper()} dataset...')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    logger.info(f'Building HRNet-W{args.width} for {args.dataset.upper()}...')
    model = HRNetCIFAR(num_classes=num_classes, width=args.width)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # Topological analysis
    topology_trainer = None
    topology_analyzer = None
    if args.topology_weight > 0:
        logger.info('Enabling topological analysis and regularization')
        topology_trainer = TopologyAwareTraining(topology_weight=args.topology_weight)
        topology_analyzer = TopologicalAnalyzer()

    # Tensorboard writer
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            logger.info(f'Resumed from epoch {start_epoch}')

    # Training loop
    logger.info('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc, train_topo_loss = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device,
            topology_trainer=topology_trainer, writer=writer
        )

        # Validate
        val_topo_stats = None
        if epoch % args.topology_interval == 0 and topology_analyzer is not None:
            val_loss, val_acc, val_topo_stats = validate(
                model, val_loader, criterion, device, topology_analyzer=topology_analyzer
            )
        else:
            val_loss, val_acc, _ = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Log epoch statistics
        logger.info(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                    f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')

        if topology_trainer:
            logger.info(f'  Topological Loss={train_topo_loss:.4f}')

        # Tensorboard logging
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('train/epoch_acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if val_topo_stats:
            writer.add_scalar('topology/persistence_entropy',
                              val_topo_stats['persistence_entropy'], epoch)
            for i, betti in enumerate(val_topo_stats['betti_numbers']):
                writer.add_scalar(f'topology/betti_{i}', betti, epoch)

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'args': args
        }

        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')
        if is_best:
            torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
            logger.info(f'New best accuracy: {best_acc:.2f}%')

    logger.info('Training completed!')
    logger.info(f'Best validation accuracy: {best_acc:.2f}%')

    writer.close()


if __name__ == '__main__':
    main()
