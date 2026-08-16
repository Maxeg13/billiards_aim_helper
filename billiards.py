import numpy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import requests
import threading
import time
import torchvision.io as io
from geom_tools import *

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

to_radians = np.pi/180
to_degrees = 1/to_radians
color_channels_num = 3

gravity_init = [8.386473, 0., 6.984284]
gravity = gravity_init
width_crop_k = 0.23
pixels_per_pitch = 850
pitch_per_pixels = 1./pixels_per_pitch

use_stream_out = True
# use_stream_out = False

# use_cap = False
use_cap = True

frame_src_path = 'data/20260812_144025.jpg'
# video_path = "data/20260811_135338.mp4"
cam_url = "http://192.168.1.201:8080"
# cam_url = "http://10.177.237.83:8080"
video_path = "data/20260811_173328.mp4"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cue = Line(createP([732, 654]), createP([702, 567]))

# Callback function to capture mouse movement
def track_mouse_coords(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Prints live coordinates in your terminal
        # print(f"X: {x}, Y: {y}", end="\r")
        cue.p1 = createP([x, y])
    # if event == cv2.EVENT_LBUTTONUP:

cv2.namedWindow("Image Window")
cv2.setMouseCallback("Image Window", track_mouse_coords)

def set_weight(conv, kern, idx):
    mean = torch.sum(kern, dim=[1,2]) / (kern.shape[1] * kern.shape[2])
    for i in range(color_channels_num):
        kern[i] -= mean[i]
        std = torch.std(kern[i])
        kern[i] /= (std ** 2)
    with torch.no_grad():
        conv.weight[idx].copy_(kern)

class PocketNet(nn.Module):
    kern_size_torch = (50, 100)
    kern_size = [kern_size_torch[1], kern_size_torch[0]]
    def __init__(self):
        super().__init__()
        self.conv_pos = nn.Conv2d(color_channels_num, 2, PocketNet.kern_size_torch)
        self.conv_neg = nn.Conv2d(color_channels_num, 1, PocketNet.kern_size_torch)
    def forward(self, x):
        return [self.conv_pos(x), self.conv_neg(x)]

class CueBaseNet(nn.Module):
    kern_size_torch = (20, 70)
    kern_size = [kern_size_torch[1], kern_size_torch[0]]
    def __init__(self):
        super().__init__()
        self.conv_pos = nn.Conv2d(color_channels_num, 2, CueBaseNet.kern_size_torch)
        # self.conv_neg = nn.Conv2d(color_channels_num, 1, PocketNet.kern_size_torch)
    def forward(self, x):
        return self.conv_pos(x)

pocket_net = PocketNet().to(device)
cue_base_net = CueBaseNet().to(device)
# pocket_net.train()
# pocket_kern1 = io.read_image("data/pocket_kern1.jpg").to(torch.float32).to(device).detach()
# pocket_kern2 = io.read_image("data/pocket_kern2.jpg").to(torch.float32).to(device).detach()
cue_base_kern1 = io.read_image("data/cue_base_kern1.jpg").to(torch.float32).to(device).detach()
cue_base_kern2 = io.read_image("data/cue_base_kern2.jpg").to(torch.float32).to(device).detach()
# kern3 = io.read_image("data/3.jpg").to(torch.float32).to(device).detach()

pocket_kern_neg = io.read_image("data/neg_1.jpg").to(torch.float32).detach()

# pocket_net.conv_pos.weight.detach()
# set_weight(pocket_net.conv_pos, pocket_kern1, 1)
# set_weight(pocket_net.conv_pos, pocket_kern2, 0)
# set_weight(pocket_net.conv_pos, kern3, 2)

set_weight(cue_base_net.conv_pos, cue_base_kern1, 0)
set_weight(cue_base_net.conv_pos, cue_base_kern2, 1)

set_weight(pocket_net.conv_neg, pocket_kern_neg, 0)
# pocket_net.conv_pos
# set_weight(pocket_net, kern3, 2)
# set_weight(pocket_net, kern4, 3)

# pocket_net.eval()

if use_stream_out:
    from flask import Flask, Response

    app = Flask("flask")
    def flask_generate_frames():
        while True:
            # Encode the frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', frame_out)
            if not ret:
                continue

            # Convert the encoded frame to bytes
            frame_bytes = buffer.tobytes()

            # Yield the frame in the MJPEG multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @app.route('/video')
    def video():
        # Return the response generated along with the specific media type (mime type)
        return Response(flask_generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        # Simple HTML page embedding the MJPEG stream
        return '<h1>OpenCV MJPEG Stream</h1><img src="/video" width="640">'

    def startFlask():
        app.run(host='0.0.0.0', port=5000, threaded=True)

    flaskThread = threading.Thread(target=startFlask, args=())
    flaskThread.start()

# 1. Open the video file
if use_cap:
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(cam_url + "/video")

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

def get_coords(frame, torch_from_model):
    max = torch.max(torch_from_model[0][0])
    idx_ij_pos = (torch_from_model[0][0] == max).nonzero()
    neg = torch_from_model[1][0, 0, idx_ij_pos[0, 1].item(), idx_ij_pos[0, 2].item()]
    # print(f"neg: {neg}")
    if neg > 4300:
        # print(f"zero out negative: {idx_ij_pos}")
        i, j = idx_ij_pos[0, 1].item(), idx_ij_pos[0, 2].item()
        torch_from_model[0][0, :, i-9:i+9, j-9:j+9] *= 0.
        max = torch.max(torch_from_model[0][0] )
        idx_ij_pos = (torch_from_model[0][0] == max).nonzero()
        # print(f"zero out negative, result: {idx_ij_pos}")

    # _______drawing
    # print(f"max: {max}, idx: {idx_ij_pos[0, 0].item()}")
    top_left = (idx_ij_pos[0, 2].item(), idx_ij_pos[0, 1].item())       # (x1, y1)
    return top_left

def extract_circles(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        image=gray_roi,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=90,
        param2=50,
        minRadius=30,
        maxRadius=60
    )

    if circles is None:
        circles = []

    return circles

def draw_ellipses(roi, ellipses, main_pitch):
    # Convert coordinates and radii to integers
    ellipses = np.uint16(np.around(ellipses))

    for i, ellipse in enumerate(ellipses):
        center = ellipse[0:2]  # (x, y)
        major_radius, minor_radius = ellipse[2:4]

        # Draw the outer circle outline (green)
        cv2.ellipse(roi, center, axes = (major_radius, minor_radius), angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        # Draw the center point (red)
        color = RED
        if i == 0: color = WHITE
        cv2.circle(roi, center, radius=2, color=color, thickness=3)


def update_gravity():
    global gravity
    while True:
        response = requests.get(cam_url + "/sensors.json")
        gravity = response.json()["gravity"]["data"][-1][1]
        time.sleep(0.4)

gravity_thread = threading.Thread(target=update_gravity, args=())
if use_cap:
    gravity_thread.start()

def find_phantom(target_ellipse):
    phantom_center = None
    target_center = target_ellipse[0:2]
    major_r, minor_r = target_ellipse[2:4]
    dist = 4
    for alpha in np.arange(0, np.pi, 0.007):
        x_off = major_r * np.cos(alpha) * 2
        y_off = minor_r * np.sin(alpha) * 2
        center = target_center.copy()
        center[0] += x_off
        center[1] += y_off

        # compensate diff btw src and roi widths
        cue_shifted = cue.addP(-createP([frame_shape[1] * width_crop_k, 0]))
        dist_tmp = signedDistP(createP(center), cue_shifted)
        # print(f"dist: {dist}")
        if abs(dist_tmp) < abs(dist):
            print()
            dist = dist_tmp
            phantom_center = center
    return phantom_center

def circles_to_ellipses(circles, main_pitch):
    ellipses = []
    for circle in circles:
        center = (circle[0], circle[1])  # (x, y)
        radius = circle[2]

        # Draw the outer circle outline (green)
        angle_shift = (int(center[1]) - frame_shape[0] // 2) * pitch_per_pixels
        # print(f"angle shift: {angle_shift}")
        minor_radius = int(radius * abs(np.sin(main_pitch + angle_shift)))
        # circle.append(minor_radius)
        ellipses.append(np.append(circle, minor_radius))
    return ellipses

# Main routine condition
def make_stay_cond():
    stay_ctr = 0
    def func():
        nonlocal stay_ctr
        if use_cap:
            return cap.isOpened()
        else:
            stay_ctr+=1
            # print(stay_ctr)
            return True
    return func

stay_cond = make_stay_cond()

# Main routine
while stay_cond():
    if use_cap:
        ret, frame_src = cap.read()

        # If ret is False, the video has reached the end
        if not ret:
            print("End of video file or cannot read the frame.")
            break
    else:
        frame_src = cv2.imread(frame_src_path)

    frame = cv2.resize(frame_src, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

    # compense roll
    roll = numpy.atan(gravity[1] / gravity[0] + 1.111e-8) / np.pi * 180
    frame_shape = frame.shape[0:2]
    center = (frame_shape[1] // 2, frame_shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -roll, scale = 1.)
    frame = cv2.warpAffine(frame, rotation_matrix, (frame_shape[1], frame_shape[0]))

    main_pitch = numpy.atan(gravity[2] / gravity[0] + 1.111e-8) + 0.01
    horizont = int(frame_shape[0]//2 - main_pitch * pixels_per_pitch)

    roi_offset_y = min(frame_shape[0] // 2, max(horizont, 0))
    roi = frame[roi_offset_y:, int(frame_shape[1] * width_crop_k) : int(frame_shape[1] * (1 - width_crop_k))]

    circles = extract_circles(roi)
    if len(circles):
        circles = circles[0]
    circles = sorted(circles, key=lambda item: item[2])
    ellipses = circles_to_ellipses(circles, main_pitch)

    roi_torch = torch.tensor(roi.transpose(2, 0, 1), dtype=torch.float32, device=device)
    roi_torch = roi_torch.unsqueeze(0)
    # pockets_torch = pocket_net(roi_torch)
    # cue_base_torch = cue_base_net(roi_torch)

    #____DRAWING BEGINGS
    # pocket_coords_A = get_coords(roi, pockets_torch)
    # pocket_coords = [int(pocket_coords_A[i] + PocketNet.kern_size[i] * 0.5) for i in range(2)]
    # pocket_coords_B = [pocket_coords_A[i] + PocketNet.kern_size[i] for i in range(2)]
    # cv2.rectangle(roi, pocket_coords_A, pocket_coords_B, GREEN, thickness=3)
    # get_coords(roi, pockets_torch)

    #____balls
    draw_ellipses(roi, ellipses, main_pitch)

    #____phantom ball
    phantom_center = None
    if len(ellipses) == 2:
        target_ellipse = ellipses[0]
        target_center = target_ellipse[0:2]
        major_r, minor_r = target_ellipse[2:4]
        phantom_center = find_phantom(target_ellipse)

    if phantom_center is not None:
        phantom_center = np.uint16(np.around(phantom_center))
        cv2.ellipse(roi, phantom_center, axes = (int(major_r), int(minor_r)), angle=0, startAngle=0, endAngle=360, color=GREEN, thickness=1, lineType=cv2.LINE_AA)

    # horizont
    cv2.line(frame, (0, horizont), (frame_shape[1], horizont), GREEN, thickness=1, lineType=cv2.LINE_AA)
    # cue
    # cv2.line(frame, cue.p1, cue.p2, GREEN, thickness=1, lineType=cv2.LINE_AA)

    # target traj
    if phantom_center is not None:
        target_center = np.uint16(np.around(target_center))
        vector = createP(target_center) - createP(phantom_center)
        vector *= 8
        vector = createP(target_center) + vector
        cv2.line(roi, phantom_center, vector, RED, thickness=1, lineType=cv2.LINE_AA)

    frame_out = frame.copy()
    cv2.imshow("Image Window", frame_out)

    if use_cap:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(40)
if use_cap:
    cap.release()
    cv2.destroyAllWindows()

# ______small analisys if needed
# Get the kernel weights
kernel = pocket_net.conv_pos.weight.to("cpu").detach().numpy()[:, 0]
# To get it as a NumPy array (requires detaching from the graph)
# kernel_numpy = conv.weight.detach().cpu().numpy()[0]
kernel_numpy = kernel
x = np.arange(kernel_numpy.shape[2])
y = np.arange(kernel_numpy.shape[1])
X, Y = np.meshgrid(x, y)

# Plot the surface
fig = plt.figure()
# fig, (ax1, ax2) = plt.subplots(1, 2)
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')
surf1 = ax1.plot_surface(X, Y, kernel[0], cmap='viridis')
surf2 = ax2.plot_surface(X, Y, kernel[0], cmap='viridis')
plt.tight_layout() # Adjusts spacing
plt.show()