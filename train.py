import os
import time
import shutil
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
# from easydict import EasyDict as edict
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from utils import *
from arg import parser
from dataset.dataset import *
from dataset.transforms import *

def main():
    train_dataloader = torch.utils.data.DataLoader(
                        VideoDataset('custom', 'train', '1',
                            transform=torchvision.transforms.Compose([
                                GroupRandomHorizontalFlip(),
                                GroupColorJittering(brightness=0.5, contrast=0.5, saturation=0, hue=0),
                                GroupMultiScaleCrop(256),
                                GroupScale(256),
                                ToTorchFormatTensor(True)])),
                        batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=True)
    # val_dataloader = torch.utils.data.DataLoader(
    #                     VideoDataset('custom', 'val', '1',
    #                         transform=torchvision.transforms.Compose([
    #                             GroupCenterCrop(256),
    #                             GroupScale(256),
    #                             ToTorchFormatTensor(True)])),
    #                     batch_size=1, shuffle=False,
    #                     num_workers=0, pin_memory=True)

    for i, p in enumerate(train_dataloader):
        print(i)
        print(p)
        break

''' 
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['MA', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'epoch{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders([args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)])

    model = None

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    # Check with Rishubh if any changes are to be made
    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)


    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Apply Normalise Transform
    # if args.modality != 'RGBDiff':
    #     normalize = GroupNormalize(input_mean, input_std)
    # else:
    #     normalize = IdentityTransform()

    train_dataloader = torch.utils.data.DataLoader(
                        VideoDataset(args.dataset, 'train', '1',
                            transform=torchvision.transforms.Compose([
                                GroupRandomHorizontalFlip(),
                                GroupColorJittering(brightness=0.5, contrast=0.5, saturation=0, hue=0),
                                GroupMultiScaleCrop(256),
                                GroupScale(256),
                                ToTorchFormatTensor(True)])),
                        batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=True,
                        drop_last=True)  # prevent something not % n_GPU
    val_dataloader = torch.utils.data.DataLoader(
                        VideoDataset(args.dataset, 'val', '1',
                            transform=torchvision.transforms.Compose([
                                GroupCenterCrop(256),
                                GroupScale(256),
                                ToTorchFormatTensor(True)])),
                        batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    # for group in policies:
    #     print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_dataloader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, 
                            edict({
                                'lr_type': args.lr_type, 
                                'lr_steps': args.lr_steps,
                                'lr': args.lr,
                                'weight_decay': args.weight_decay,
                                'epochs': args.epochs
                            }))

        # train for one epoch
        train(train_dataloader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_dataloader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_prec1': best_prec1,
                            }, 
                            edict({
                                    'root_model': args.root_model, 
                                    'store_name': args.store_name,
                                    # 'save_every_ckpt': args.save_every_ckpt
                                }),
                            is_best)


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg

'''
if __name__ == '__main__':
    main()