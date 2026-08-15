import numpy
import torch
import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import requests
# from PIL.ImageOps import scale
import threading
import time

RED = (0, 0, 255)
WHITE = (255, 255, 255)

to_radians = np.pi/180
to_degrees = 1/to_radians

gravity_init = [8.386473, 0., 6.984284]
gravity = gravity_init
width_crop_k = 0.23
pixels_per_pitch = 591
pitch_per_pixels = 1./pixels_per_pitch

use_cap = False
# use_cap = True
# video_path = "data/20260811_135338.mp4"
cam_url = "http://192.168.1.201:8080"
# cam_url = "http://10.177.237.83:8080"
video_path = "data/20260811_173328.mp4"

if torch.cuda.is_available():
    device = torch.device("cuda")

import torchvision.io as io

class PocketNet(nn.Module):
    kern_size = (50, 100)
    def __init__(self):
        super().__init__()
        self.conv_pos = nn.Conv2d(3, 2, PocketNet.kern_size)
        self.conv_neg = nn.Conv2d(3, 1, PocketNet.kern_size)
    def forward(self, x):
        return [self.conv_pos(x), self.conv_neg(x)]


pocketNet = PocketNet().to(device)
# pocketNet.train()
pocket_kern1 = io.read_image("data/pocket_kern1.jpg").to(torch.float32).to(device).detach()
pocket_kern2 = io.read_image("data/pocket_kern2.jpg").to(torch.float32).to(device).detach()
# kern3 = io.read_image("data/3.jpg").to(torch.float32).to(device).detach()

pocket_kern_neg = io.read_image("data/neg_1.jpg").to(torch.float32).detach()

def set_weight(conv, kern, idx):
    mean = torch.sum(kern, dim=[1,2]) / (kern.shape[1] * kern.shape[2])
    for i in range(3):
        kern[i] -= mean[i]
        std = torch.std(kern[i])
        kern[i] /= (std ** 2)
    with torch.no_grad():
        conv.weight[idx].copy_(kern)

# pocketNet.conv_pos.weight.detach()
set_weight(pocketNet.conv_pos, pocket_kern1, 1)
set_weight(pocketNet.conv_pos, pocket_kern2, 0)
# set_weight(pocketNet.conv_pos, kern3, 2)

set_weight(pocketNet.conv_neg, pocket_kern_neg, 0)
# pocketNet.conv_pos
# set_weight(pocketNet, kern3, 2)
# set_weight(pocketNet, kern4, 3)

# pocketNet.eval()


# 1. Open the video file
if use_cap:
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(cam_url + "/video")

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
else:
    frame = cv2.imread('data/20260812_144025.jpg')

def draw_rect(frame, torch_from_model, className):
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
    bottom_right = (idx_ij_pos[0, 2].item() + className.kern_size[1], idx_ij_pos[0, 1].item() + className.kern_size[0])  # (x2, y2)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

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
    return circles

def draw_ellipses(roi, ellipses, main_pitch):
    if ellipses is not None:
        # Convert coordinates and radii to integers
        ellipses = np.uint16(np.around(ellipses))

        for i, ellipse in enumerate(ellipses):
            center = (ellipse[0], ellipse[1])  # (x, y)
            radius = ellipse[2]
            minor_radius = ellipse[3]

            # Draw the outer circle outline (green)
            cv2.ellipse(roi, center, axes = (radius, minor_radius), angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=1)

            # Draw the center point (red)
            color = RED
            if i == 0: color = WHITE
            cv2.circle(roi, center, 2, color, 3)


def update_gravity():
    global gravity
    while True:
        response = requests.get(cam_url + "/sensors.json")
        gravity = response.json()["gravity"]["data"][-1][1]
        time.sleep(0.4)

gravity_thread = threading.Thread(target=update_gravity, args=())
if use_cap:
    gravity_thread.start()

def circles_to_ellipses(circles, main_pitch):
    ellipses = []
    for circle in circles:
        center = (circle[0], circle[1])  # (x, y)
        radius = circle[2]

        # Draw the outer circle outline (green)
        angle_shift = (int(center[1]) - frame_shape[0] // 2) * pitch_per_pixels
        print(f"angle shift: {angle_shift}")
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
            return stay_ctr < 2
    return func

stay_cond = make_stay_cond()

# Main routine
while stay_cond():
    if use_cap:
        ret, frame = cap.read()

        # If ret is False, the video has reached the end
        if not ret:
            print("End of video file or cannot read the frame.")
            break

    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

    # compense roll
    roll = numpy.atan(gravity[1] / gravity[0] + 1.111e-8) / np.pi * 180
    frame_shape = frame.shape[0:2]
    center = (frame_shape[1] // 2, frame_shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -roll, scale = 1.)
    frame = cv2.warpAffine(frame, rotation_matrix, (frame_shape[1], frame_shape[0]))

    roi = frame[:, int(frame_shape[1] * width_crop_k) : int(frame_shape[1] * (1 - width_crop_k))]

    main_pitch = numpy.atan(gravity[2] / gravity[0] + 1.111e-8)
    circles = extract_circles(roi)
    if circles is not None:
        circles = circles[0]
    circles = sorted(circles, key=lambda item: item[2])
    ellipses = circles_to_ellipses(circles, main_pitch)

    roi_torch = torch.tensor(roi.transpose(2, 0, 1), dtype=torch.float32, device=device)
    roi_torch = roi_torch.unsqueeze(0)
    pockets_torch = pocketNet(roi_torch)

    #____drawing
    horizont = int(frame_shape[0]//2 - main_pitch * pixels_per_pitch)

    draw_rect(roi, pockets_torch, PocketNet)
    draw_ellipses(roi, ellipses, main_pitch)

    cv2.line(frame, (0, horizont), (frame_shape[1], horizont), (0, 255, 0), thickness=1)

    cv2.imshow("Video Playback", frame)

    if use_cap:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
if use_cap:
    cap.release()
    cv2.destroyAllWindows()

# ______small analisys if needed
# Get the kernel weights
kernel = pocketNet.conv_pos.weight.to("cpu").detach().numpy()[:, 0]
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