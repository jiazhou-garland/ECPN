import os, torch, datetime, time
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from loss import cross_entropy_loss_RCF, distancemap_penalized_loss, mixed_loss
from data.max_dataloader import CustomImageDataset
import torch.backends.cudnn as cudnn
from model.model import MLEEP

if __name__ == '__main__':

    # training parameters-----------------------------------------------------------------------------------------------
    batch_size = 8
    epochs = 120
    early_stopping = 20
    # optimizer
    lr = 1e-6
    min_lr = 1e-9
    weight_decay = 2e-4
    # data
    num_workers = 20
    num_workers_val = 0
    # log
    experiment = 'EXP_num'
    InteLog = 10
    # Prepare saving model--------------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_path = os.path.join('../runs', experiment)
    if not os.path.exists(current_path): os.mkdir(current_path)
    model_dir = os.path.join(current_path, 'models')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    log_dir = os.path.join(current_path, 'logs')
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    # Prepare model-----------------------------------------------------------------------------------------------------------
    device_num = 2
    device = torch.device("cuda:" + str(device_num))
    model = MLEEP(multi_inputs=False, upsample_type='Transpose-conv', decoder_type='Bidecoder')
    model.to(device)

    # Load model------------------------------------------------------------------------------------------------------
    # model_ckpt = "model.pth"
    # model.load_state_dict(torch.load(model_ckpt))

    # prepare data --------------------------------------------------------------------------------------------------
    trainTxtPath = r'max_train.txt'
    datasets = CustomImageDataset(trainTxtPath, aug=False)
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available(),
                        drop_last=True,
                        num_workers=num_workers)

    valTxtPath = r'max_val.txt'
    datasets_val = CustomImageDataset(valTxtPath, aug=False, )
    valid_loader = DataLoader(datasets_val, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
                            drop_last=False, num_workers=num_workers_val)
    STEPS_val = len(valid_loader)

    # Optimizer and lr_scheduler----------------------------------------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_adam = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=2e-4)
    optimizer_RMSprop = optim.RMSprop(params, lr=lr, weight_decay=0.9, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_adam, T_0=3, T_mult=2,
                                                                        eta_min=min_lr,last_epoch=-1)
    # Loss function-----------------------------------------------------------------------------------------------------------
    loss1 = cross_entropy_loss_RCF
    loss2 = distancemap_penalized_loss
    loss3 = mixed_loss
    side_weight = 0.5
    fuse_weight = 1.1
    # Metrics-----------------------------------------------------------------------------------------------------------
    IOU = smp.utils.metrics.IoU(threshold=0.5)
    Fscore = smp.utils.metrics.Fscore()

    # train the model --------------------------------------------------------------------------------------------------
    def train(dataloader, model, loss_, optimizer, IOU, Fscore, epoch, cuda, supervision):
        model.train()  # use batch normalization and drop out

        size = len(dataloader.dataset)
        running_loss, running_iou, running_Fscore = 0.0, 0.0, 0.0
        end_batch = time.time()

        for step, (batch_x, batch_y) in enumerate(dataloader):

            batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # compute output
            output = model(batch_x)
            loss = torch.zeros(1).to(device)
            if supervision:
                for k in range(5):
                    loss += side_weight * loss_(output[k], batch_y, cuda=cuda) / batch_size
                loss += fuse_weight * loss_(output[-1], batch_y, cuda=cuda) / batch_size
            else:
                loss = loss_(output[-1].to(device), batch_y, cuda=cuda)

            iou = IOU(torch.sigmoid(output[-1]).to(device), batch_y)
            fscore = Fscore(torch.sigmoid(output[-1]).to(device), batch_y)

            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_iou += iou.item()
            running_Fscore += fscore.item()

            if step % InteLog == 0:

                writer.add_scalar('training loss', running_loss / 10, epoch * size + step)
                writer.add_scalar('training iou', running_iou / 10, epoch * size + step)
                writer.add_scalar('training Fscore', running_Fscore / 10, epoch * size + step)

                running_loss = 0.0
                running_iou = 0.0
                running_Fscore = 0.0

            current = step * len(batch_x)
            batch_time = time.time() - end_batch
            end_batch = time.time()
            print({f"batch time: {batch_time:.3f} loss: {loss.item():>7f} iou: {iou.item():>7f} Fsorce:{fscore.item():>7f}"
                   f" [current{current:>5d}/batch_size{size:>5d}]"})

        # test the model ---------------------------------------------------------------------------------------------------

    def validate(dataloader, model, loss_, IOU, Fscore, epoch, cuda, supervision):
        model.eval()  # switch to evaluate mode

        size = len(dataloader.dataset)
        mean_loss, mean_iou, mean_Fscore = 0.0, 0.0, 0.0
        end = time.time()

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                output = model(batch_x)

                loss = torch.zeros(1).to(device)
                if supervision:
                    for k in range(5):
                        loss += side_weight * loss_(output[k], batch_y, cuda=cuda) / batch_size
                    loss += fuse_weight * loss_(output[-1], batch_y, cuda=cuda) / batch_size
                else:
                    loss = loss_(output[-1].to(device), batch_y, cuda=cuda)

                iou = IOU(torch.sigmoid(output[-1]).to(device), batch_y)
                fscore = Fscore(torch.sigmoid(output[-1]).to(device), batch_y)

                mean_loss += loss.item()
                mean_iou += iou.item()
                mean_Fscore = fscore.item()

                print({f"current_loss: {loss.item():>7f} current_iou: {iou.item():>7f} "
                       f"current_Fscore{fscore.item():>7f}"})

        mean_loss /= size
        mean_iou /= size
        mean_Fscore /= size

        writer.add_scalar('val_loss', mean_loss, epoch)
        writer.add_scalar('val_iou', mean_iou, epoch)
        writer.add_scalar('val_Fscore', mean_Fscore, epoch)

        val_time = time.time() - end
        print(f"Test Error: \n Mean_IOU: {(100 * mean_iou):>0.1f}%, Mean_Loss: {mean_loss:>8f} "
              f"Mean_Fscore: {mean_Fscore:>8f}\n Total val time: {val_time:.3f}")

        return mean_loss, mean_iou, mean_Fscore

    # train and validate in epochs--------------------------------------------------------------------------------------

    cudnn.benchmark = True

    best_loss = np.Inf
    best_iou = 0.0
    best_Fscore = 0.0
    trigger = 0
    end_epoch = time.time()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss1, optimizer, IOU, Fscore, epoch=t, cuda=device_num, supervision=True)
        mean_loss, mean_iou, mean_Fscore = validate(valid_loader, model, loss1, IOU, Fscore, epoch=t,
                                                    cuda=device_num, supervision=False)
        writer.add_scalar('lr', optimizer_adam.param_groups[0]['lr'], t)
        lr_scheduler.step\
            ()
        epoch_time = time.time() - end_epoch
        print(f"the {t + 1}th epoch_time: {epoch_time:.3f}")
        end_epoch = time.time()
        trigger += 1

        if mean_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_loss_model.pth'))
            best_loss = mean_loss
            print('=> saved best loss model to ' + os.path.join(model_dir, 'best_loss_model.pth'))
            trigger = 0

        if mean_iou > best_iou:
            torch.save(model.state_dict(), os.path.join(model_dir, 'MLEEP(multi-inputs).pth'))
            best_iou = mean_iou
            print('=> saved best Iou model to ' + os.path.join(model_dir, 'MLEEP(multi-inputs).pth'))
            trigger = 0

        if mean_Fscore > best_Fscore:
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_F-score_model.pth'))
            best_Fscore = mean_Fscore
            print('=> saved best F-score model to ' + os.path.join(model_dir, 'best_F-score_model.pth'))
            trigger = 0

        # if trigger >= early_stopping:
        #     print("=> early stopping at the " + str(t+1) + "th epochs")
        #     break

        torch.cuda.empty_cache()

    print("All is Done! The best model's loss is " + str(best_loss))