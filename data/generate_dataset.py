import os
def sort_images(image_dir, image_type):
    """
    sort images in the folder based on their names

    image_dir：
    image_type：jpg or npy
    """
    files = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
                and not image_name.startswith('.'):
            files.append(os.path.join(image_dir, image_name))

    return sorted(files)

def write_file(mode, images, labels):
    """
    save the images and labels into a __.txt file
    """
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(labels)):
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(images[8*i], images[8*i+1], images[8*i+2],images[8*i+3],
                                                                  images[8*i+4],images[8*i+5],images[8*i+6],images[8*i+7],
                                                                  labels[i]))

def train_val_divide(image_dir,label_dir, val_ratio=0.2, fold=0):
    """
    divide the train ana validation data,

    val_ratio = 0.8
    fold = 0
    return: __.txt
            --> Tuple[List[Tuple["image_name","label_name"], List[Tuple["val_image_name","val_label_name"]]]
    """

    print('Begin to divide train and val')
    images = sort_images(image_dir, 'jpg')
    labels = sort_images(label_dir, 'png')

    # print(len(images))
    # print(len(labels))
    train_im = []
    train_label = []
    val_im = []
    val_label = []
    val_interval = int((1 / val_ratio))
    for i in range(len(labels)):
        if ((i+1) == fold or (i+1)%val_interval == fold):
            for j in range(8):
                val_im.append(images[8*i+j])
            val_label.append(labels[i])
        else:
            for j in range(8):
                train_im.append(images[8*i+j])
            train_label.append(labels[i])

    write_file('train', train_im, train_label)
    write_file('val', val_im, val_label)
    print("Done!")

def max_train_val_divide(max_image_dir, label_dir):

    print('Begin to divide train and val')
    images = sort_images(max_image_dir, 'jpg')
    labels = sort_images(label_dir, 'png')

    train_im = []
    train_label = []
    val_im = []
    val_label = []
    m, n = 0, 0

    for i in range(len(labels)):
        if m == 4:
            val_im.append(images[i])
            val_label.append(labels[i])
            m = 0
        else:
            train_im.append(images[i])
            train_label.append(labels[i])
            m += 1

    with open('./{}.txt'.format('max_train'), 'w') as f:
        for i in range(len(train_label)):
            f.write('{}\t{}\n'.format(train_im[i], train_label[i]))

    with open('./{}.txt'.format('max_val'), 'w') as f:
        for i in range(len(val_label)):
            f.write('{}\t{}\n'.format(val_im[i], val_label[i]))

    print("Done!")

def whole_dataset(image_dir, label_dir):

    images = sort_images(image_dir, 'jpg')
    labels = sort_images(label_dir, 'png')

    im = []
    label = []

    for i in range(len(labels)):
        for j in range(8):
            im.append(images[8*i+j])
        label.append(labels[i])

    write_file('whole', im, label)

def whole_max_dataset(max_image_dir, label_dir):

    images = sort_images(max_image_dir, 'jpg')
    labels = sort_images(label_dir, 'png')

    print(len(images))
    print(len(labels))
    im = []
    label = []

    for i in range(len(labels)):
        im.append(images[i])
        label.append(labels[i])

    with open('./{}.txt'.format('whole_max'), 'w') as f:
        for i in range(len(labels)):
            f.write('{}\t{}\t\n'.format(images[i], labels[i]))      

if __name__ == "__main__":
    image_dir = 'im_dir'
    label_dir = 'label_dir'
    max_image_dir = "max_image_dir"
    # train_val_divide(image_dir, label_dir, val_ratio=0.2, fold=0)
    # max_train_val_divide(max_image_dir, label_dir)
    # whole_dataset(image_dir, label_dir)
    # whole_max_dataset(max_image_dir, label_dir)

