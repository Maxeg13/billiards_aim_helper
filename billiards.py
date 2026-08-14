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

gravity_init = [8.386473, 0., 6.984284]
gravity = gravity_init
width_crop_k = 0.23
pixels_per_pitch = 5.4 / (np.pi/6)
pitch_per_pixels = 1./pixels_per_pitch
to_radians = np.pi/180
to_degrees = 1/to_radians
kern_size = (50, 100)

use_cap = False
# use_cap = True
# video_path = "data/20260811_135338.mp4"
# cam_url = "http://192.168.1.201:8080"
cam_url = "http://10.177.237.83:8080"
video_path = "data/20260811_173328.mp4"

if torch.cuda.is_available():
    device = torch.device("cuda")

import torchvision.io as io

# functions to show an image

# Initialize the transform
# num_output_channels=1 returns a single channel (L)
# num_output_channels=3 returns 3 channels where R == G == B
# grayscale_transform = transforms.Grayscale(num_output_channels=1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pos = nn.Conv2d(3, 2, kern_size)
        self.conv_neg = nn.Conv2d(3, 1, kern_size)


    def forward(self, x):
        return [self.conv_pos(x), self.conv_neg(x)]


model = Net().to(device)
# model.train()
kern1 = io.read_image("data/1.jpg").to(torch.float32).to(device).detach()
kern2 = io.read_image("data/2.jpg").to(torch.float32).to(device).detach()
# kern3 = io.read_image("data/3.jpg").to(torch.float32).to(device).detach()

kern_neg = io.read_image("data/neg_1.jpg").to(torch.float32).detach()

def set_weight(conv, kern, idx):
    mean = torch.sum(kern, dim=[1,2]) / (kern_size[0] * kern_size[1])
    for i in range(3):
        kern[i] -= mean[i]
        std = torch.std(kern[i])
        kern[i] /= (std ** 2)
    with torch.no_grad():
        conv.weight[idx].copy_(kern)

# model.conv_pos.weight.detach()
set_weight(model.conv_pos, kern1, 1)
set_weight(model.conv_pos, kern2, 0)
# set_weight(model.conv_pos, kern3, 2)

set_weight(model.conv_neg, kern_neg, 0)
# model.conv_pos
# set_weight(model, kern3, 2)
# set_weight(model, kern4, 3)

# model.eval()


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

def draw_rect(frame, torch_from_model):
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
    bottom_right = (idx_ij_pos[0, 2].item() + kern_size[1], idx_ij_pos[0, 1].item() + kern_size[0])  # (x2, y2)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

def extract_circles(frame_roi):
    gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
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

def draw_circles(frame_roi, circles, horizont):
    if circles is not None:
        # Convert coordinates and radii to integers
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])  # (x, y) coordinates of center
            radius = i[2]          # radius

            # Draw the outer circle outline (green)
            # cv2.circle(frame_roi, center, radius, (0, 255, 0), thickness=1)
            angle = (int(center[1]) - horizont) * pitch_per_pixels * to_radians
            minor_radius = int(radius * abs(np.sin(angle)))
            cv2.ellipse(frame_roi, center, axes = (radius, minor_radius), angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=1)

            # Draw the center point (red)
            cv2.circle(frame_roi, center, 2, (0, 0, 255), 3)


def update_gravity():
    global gravity
    while True:
        response = requests.get(cam_url + "/sensors.json")
        gravity = response.json()["gravity"]["data"][-1][1]
        time.sleep(0.4)

gravity_thread = threading.Thread(target=update_gravity, args=())
if use_cap:
    gravity_thread.start()

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
    shape = frame.shape[0:2]
    center = (shape[1] // 2, shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -roll, scale = 1.)
    frame = cv2.warpAffine(frame, rotation_matrix, (shape[1], shape[0]))

    pitch = numpy.atan(gravity[2] / gravity[0] + 1.111e-8) / np.pi * 180

    frame_roi = frame[:, int(shape[1] * width_crop_k) : int(shape[1] * (1 - width_crop_k))]

    circles = extract_circles(frame_roi)

    frame_torch = torch.tensor(frame_roi.transpose(2, 0, 1), dtype=torch.float32, device=device)
    frame_torch = frame_torch.unsqueeze(0)
    torch_from_model = model(frame_torch)

    #____drawing
    horizont = int(shape[0]//2 - pitch * pixels_per_pitch)
    draw_rect(frame_roi, torch_from_model)
    draw_circles(frame_roi, circles, horizont)

    cv2.line(frame, (0, horizont), (shape[1], horizont), (0, 255, 0), thickness=1)

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
kernel = model.conv_pos.weight.to("cpu").detach().numpy()[:, 0]
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