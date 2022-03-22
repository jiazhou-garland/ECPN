import os, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataloader import CustomImageDataset
import segmentation_models_pytorch as smp
from loss import cross_entropy_loss_RCF, distancemap_penalized_loss, mixed_loss
from data.max_dataloader import CustomImageDataset
from rockslice_edge_detector.model.model_CSv2 import EfficientNetV2_Bidecoder
from torch.nn import functional as F
from os.path import join, abspath, dirname
from scipy import io
from PIL import Image

def test(model, exp, device_num):

    # ================ device
    device = torch.device("cuda:" + str(device_num))

    # ================ data
    valTxtPath = r'/max_val.txt'
    test_dataset = CustomImageDataset(valTxtPath, val=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=20, drop_last=False, shuffle=False,
                             pin_memory=torch.cuda.is_available())
    # ================ make directory
    THIS_DIR = abspath(dirname(__file__))
    result_path = join(THIS_DIR, 'result')
    if not os.path.exists(result_path): os.mkdir(result_path)
    EXP_num = exp
    label_save_dir = join(result_path, EXP_num)
    if not os.path.exists(label_save_dir): os.mkdir(label_save_dir )

    # ================ prepare model
    model.to(device).eval()

    # ================ loss
    loss1 = cross_entropy_loss_RCF
    loss2 = distancemap_penalized_loss
    loss3 = mixed_loss

    # ================ evaluation
    running_loss, running_iou, running_Fscore, running_R, running_P, running_Accuracy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    IOU = smp.utils.metrics.IoU(threshold=0.5)
    Fscore = smp.utils.metrics.Fscore()
    Recall = smp.utils.metrics.Recall()
    Precision = smp.utils.metrics.Precision()
    Accuracy = smp.utils.metrics.Accuracy()
    size = len(test_loader)
    end = time.time()
    # ================ evaluation and save image and mat
    for i,  (image, label, labelpath) in enumerate(test_loader):
        image, label = image.type(torch.FloatTensor), label.type(torch.FloatTensor)
        image, label = image.to(device), label.to(device)
        output = model(image)

        loss = torch.zeros(1).to(device)
        loss += loss1(output[-1], label, cuda=device_num)
        iou = IOU(torch.sigmoid(output[-1]).to(device), label)
        fscore = Fscore(torch.sigmoid(output[-1]).to(device), label)
        P = Precision(torch.sigmoid(output[-1]).to(device), label)
        R = Recall(torch.sigmoid(output[-1]).to(device), label)
        A =  Accuracy(torch.sigmoid(output[-1]).to(device), label)

        running_loss += loss.item()  # .item()使张量转化为数字
        running_iou += iou.item()
        running_Fscore += fscore.item()
        running_P += P.item()
        running_R += R.item()
        running_Accuracy += A.item()

        output = [F.sigmoid(i) for i in output]
        result = torch.squeeze(output[-1].detach()).cpu().numpy()
        filename = labelpath[0][53:-9]
        result_image = Image.fromarray((result * 255).astype(np.uint8))
        print(join(label_save_dir, "%s.png" % filename))
        result_image.save(join(label_save_dir, "%s.png" % filename))

        print("%s Running test [%d/%d]" % (filename, i + 1, len(test_loader)))

    mean_loss = running_loss / size
    mean_iou = running_iou / size
    mean_Fscore = running_Fscore / size
    mean_P = running_P / size
    mean_R = running_R / size
    mean_A = running_Accuracy / size
    val_time = time.time() - end
    print(f"Test Error: \n Mean_IOU: {(100 * mean_iou):>0.1f}%, Mean_Loss: {mean_loss:>8f} "
          f"Mean_Fscore: {mean_Fscore:>8f}, Mean_P: {mean_P:>8f}, Mean_R: {mean_R:>8f}, Mean_A:{mean_A:>8f}\n")
    print(f"Test Total Time: {val_time:>8f}\n")

if __name__ == '__main__':
    exp_num = 'EXP_num'
    device_num = 0
    device = torch.device("cuda:" + str(device_num))
    model_path = r"model.pth"
    model = EfficientNetV2_Bidecoder(cuda_num=0, multi_inputs=False, upsample_type='Transpose-conv', decoder_type='Bidecoder')
    model.load_state_dict(torch.load(model_path, map_location=device))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.8fM" % (total ))
    test(model, exp_num, device_num)

    # save_gt_label_mat()