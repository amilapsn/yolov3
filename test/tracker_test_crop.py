import time

from utils.track_utils import initiate_tracker, detect_key_points_crop, initiate_matcher
from utils.track_utils import NewBBox, track_bboxes, print_id

from models import *
from utils.datasets import *
from utils.utils import *

cfg = '../cfg/traffic.cfg'
data_cfg = '../data/traffic.data'
weights = '../weights/best.pt'
output = '../output'
images = "../data/track_images"
img_size = 416
conf_thres = 0.5
nms_thres = 0.5

device = torch_utils.select_device()
if os.path.exists(output):
    shutil.rmtree(output)  # delete output folder
os.makedirs(output)  # make new output folder

#model
model = Darknet(cfg, img_size)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()

#data loader
dataloader = LoadImages(images, img_size=img_size)
classes = load_classes('../data/traffic.names')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

tracker = initiate_tracker(500)
bf_Matcher = initiate_matcher()
tracked_bbox_list = []
for i, (path, img, im0, vid_cap) in enumerate(dataloader):
    t = time.time()
    save_path = str(Path(output) / Path(path).name)

    img = torch.from_numpy(img).unsqueeze(0).to(device)
    pred, _ = model(img)
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

    new_bbox_list =[]

    if detections is not None and len(detections) > 0:
        # Rescale boxes from 416 to true image size
        scale_coords(img_size, detections[:, :4], im0.shape).round()

        # Print results to screen
        for c in detections[:, -1].unique():
            n = (detections[:, -1] == c).sum()
            print('%g %ss' % (n, classes[int(c)]), end=', ')

        # Draw bounding boxes and labels of detections
        for *xyxy, conf, cls_conf, cls in detections:
            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf)
            # t1 = time.time()
            kp, des = detect_key_points_crop(im0, xyxy, tracker)
            # print('Time taken for keypoints : (%.3fs)' % (time.time() - t1))

            new_bbox_list.append(NewBBox(kp, des, int(cls), float(conf), xyxy))

            # cv2.drawKeypoints(im0, kp, im0, color=(255, 0, 0))
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

    tracked_bbox_list = track_bboxes(bf_Matcher, tracked_bbox_list, new_bbox_list)
    print_id(im0, tracked_bbox_list)

    print('Time taken : (%.3fs)' % (time.time() - t))
    cv2.imwrite(save_path, im0)



