import os, time
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from loss import cross_entropy_loss_RCF, distancemap_penalized_loss, mixed_loss
from data.dataloader import CustomImageDataset
from model.model_CSv2 import MLEEP
from torch.nn import functional as F
from os.path import join, abspath, dirname
from PIL import Image

def predict_whole_label(model,cuda_num):
    # # ================ device
    device = torch.device("cuda:" + str(cuda_num))
    # ================ data
    valTxtPath = r'whole.txt'
    test_dataset = CustomImageDataset(valTxtPath, val=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=4, drop_last=False, shuffle=False,
                             pin_memory=torch.cuda.is_available())
    # # ================ make directory
    THIS_DIR = abspath(dirname(__file__))
    result_path = join(THIS_DIR, 'result')
    if not os.path.exists(result_path): os.mkdir(result_path)
    result_num = join(result_path, exp_num)
    if not os.path.exists(result_num): os.mkdir(result_num)
    label_save_dir = join(result_num, 'predicted_all_label')
    if not os.path.exists(label_save_dir): os.mkdir(label_save_dir)
    whole_label_save_dir = join(result_num, 'whole_label')
    if not os.path.exists(whole_label_save_dir): os.mkdir(whole_label_save_dir)
    # ================ prepare model
    model.to(device).eval()
    # ================ evaluation and save image and mat
    for i,  (image, label, labelpath) in enumerate(test_loader):
        image, label = image.type(torch.FloatTensor), label.type(torch.FloatTensor)
        image, label = image.to(device), label.to(device)
        output = model(image)

        output = [F.sigmoid(i) for i in output]
        result = torch.squeeze(output[-1].detach()).cpu().numpy()
        filename = labelpath[0][53:-9]
        result_image = Image.fromarray((result * 255).astype(np.uint8))
        result_image.save(join(label_save_dir, "%s.png" % filename))
        print("%s Running test [%d/%d]" % (filename, i + 1, len(test_loader)))

    folders_num = [str(i + 1) for i in range(25)]
    for i in range(9):
        folders_num[i] = "0" + folders_num[i]
    rock_num = ["38", "91", "139", "152", "167"]

    for k in range(5):
        for i in range(25):
            predicted_label_whole = np.empty([1536, 2048])
            for h in range(5):
                for w in range(7):
                    label_path = label_save_dir + "/" + rock_num[k] + "_" + folders_num[i] + "_" + "h" + str(h+1) + "w" + str(w+1) + ".png"
                    label = cv2.imread(label_path,cv2.COLOR_BGR2GRAY)
                    predicted_label_whole[h * 256:(h + 2) * 256, w * 256:(w + 2) * 256] = label
            whole_label_dir = join(whole_label_save_dir, rock_num[k] + "_" + folders_num[i] + '.png')
            print(whole_label_dir)
            cv2.imwrite(whole_label_dir, predicted_label_whole)

def test(model, exp, device_num):

    # ================ device
    device = torch.device("cuda:" + str(device_num))

    # ================ data
    valTxtPath = r'val.txt'
    test_dataset = CustomImageDataset(valTxtPath, val=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=4, drop_last=False, shuffle=False,
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
    exp_num = 'exp_num'
    device_num = 1
    device = torch.device("cuda:" + str(device_num))
    model_path = r"MLEEP(multi-inputs).pth"
    model = MLEEP(cuda_num=device_num, multi_inputs=True, decoder_type="Bidecoder", upsample_type='Transpose-conv')
    model.load_state_dict(torch.load(model_path, map_location=device))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.8f" % (total ))
    test(model, exp_num, device_num)
    predict_whole_label(model, cuda_num=device_num)


