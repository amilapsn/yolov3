import time

from utils.track_utils import initiate_tracker, assign_keypoints, initiate_matcher
from utils.track_utils import NewBBox, track_bboxes, generate_mask, detect_kp2

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

tracker = initiate_tracker(2000)
bf_Matcher = initiate_matcher()
tracked_bbox_list = []
for i, (path, img, im0, vid_cap) in enumerate(dataloader):
    t = time.time()
    save_path = str(Path(output) / Path(path).name)

    img = torch.from_numpy(img).unsqueeze(0).to(device)
    pred, _ = model(img)
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

    new_bbox_list =[]
    img_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    t2 = time.time()
    print('Time taken for detection : (%.3fs)' % (t2 - t))
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
            mask = generate_mask(mask,xyxy)
            # print('Time taken for keypoints : (%.3fs)' % (time.time() - t1))

            new_bbox_list.append(NewBBox.no_keypoints(int(cls), float(conf), xyxy))

            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

    kp, des = detect_kp2(img_gray,mask,tracker)
    cv2.drawKeypoints(im0, kp, im0, color=(255, 0, 0))
    t3 = time.time()
    print('Time taken for key point detection : (%.3fs)' % (t3 - t2))
    assign_keypoints(new_bbox_list, kp, des)
    print('Time taken for assigning key points : (%.3fs)' % (time.time() - t3))
    tracked_bbox_list = track_bboxes(bf_Matcher, tracked_bbox_list, new_bbox_list)

    print('Time taken : (%.3fs)' % (time.time() - t))
    cv2.imwrite(save_path, im0)
    # for tbb in tracked_bbox_list:
    #     tbb.print_details()


