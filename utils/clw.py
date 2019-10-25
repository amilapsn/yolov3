
"""calculates class weight of the dataset to be given to the Cross Entropy Loss
function that get the class confidence loss
"""
import torch

def f(train,names):
    n = len(open(names).readlines())
    ims = open(train).readlines()
    txts = [im.replace('jpg','txt').strip().replace('images','labels') for im in ims]
    class_weight = [0] * n

    for i,txt in enumerate(txts):
        objects = open(txt).readlines()
        objects = [int(obj.strip().split()[0]) for obj in objects]
        for obj in objects:
            class_weight[obj] += 1

    s = sum(class_weight)
    class_weight = [w/s for w in class_weight]
    class_weight = torch.FloatTensor(class_weight).cuda()
    return class_weight

if __name__ == '__main__':
    from sys import argv
    print(f(argv[1],argv[2]))
