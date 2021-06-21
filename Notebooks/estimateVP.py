#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:53:47 2021

@author: nitinpabreja
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from skimage import feature, color, transform, io
import logging

class VanishingPoints:
  """
  Estimate vanishing points for a video
  """

  def __init__(self, video):
    self.video = video

  def klt_tracker(self, video):
    cap = cv2.VideoCapture(video)
    # params for corner detection
    feature_params = dict( maxCorners = 150,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                             **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame_count=0
    img = []

    while(frame_count < 100):
      frame_count +=1
      cap.set(cv2.CAP_PROP_FPS,12.5) 
      ret, frame = cap.read()
      frame_gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)
    
      # calculate optical flow
      p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                            frame_gray,
                                            p0, None,
                                            **lk_params)
    
      # Select good points
      good_new = p1[st == 1]
      good_old = p0[st == 1]
    
      # draw the tracks
      for i, (new, old) in enumerate(zip(good_new, 
                                        good_old)):
          a, b = new.ravel()
          c, d = old.ravel()
          mask = cv2.line(mask, (a, b), (c, d),
                          color[i].tolist(), 2)
            
          frame = cv2.circle(frame, (a, b), 5,
                            color[i].tolist(), -1)
            
      img = cv2.add(frame, mask)
    
      # Updating Previous frame and points 
      old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1, 1, 2)
    
    cv2.destroyAllWindows()
    cap.release()
    return img

  def compute_edgelets(self, image, sigma=3):
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=3,
                                               line_gap=2)

    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)

  def vis_edgelets(self, image, edgelets, show=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap = 'gray')
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]

        plt.plot(xax, yax, 'r-')

    if show:
        plt.show()

  def edgelet_lines(self, edgelets):
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

  def compute_votes(self, edgelets, model, threshold_inlier=5):
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths

  def ransac_vanishing_point(self, edgelets, num_ransac_iter=2000, threshold_inlier=5):
    locations, directions, strengths = edgelets
    lines = self.edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = self.compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_model

  def reestimate_model(self, model, edgelets, threshold_reestimate=5):
    locations, directions, strengths = edgelets

    inliers = self.compute_votes(edgelets, model, threshold_reestimate) > 0
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = self.edgelet_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    est_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((est_model, [1.]))

  def vis_model(self, image, model, show=True):
    import matplotlib.pyplot as plt
    edgelets = self.compute_edgelets(image)
    locations, directions, strengths = edgelets
    inliers = self.compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    self.vis_edgelets(image, edgelets, False)
    vp = model / model[2]
    plt.plot(vp[0], vp[1], 'bo')
    for i in range(locations.shape[0]):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        plt.plot(xax, yax, 'b-.')

    if show:
        plt.show()
    
  def remove_inliers(self, model, edgelets, threshold_inlier=10):
    inliers = self.compute_votes(edgelets, model, threshold_inlier) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets

  def first_vp(self):
    image = self.klt_tracker(self.video)
    edgelets1 = self.compute_edgelets(image)
    vp1 = self.ransac_vanishing_point(edgelets1,num_ransac_iter=2000, threshold_inlier=5)
    vp1 = self.reestimate_model(vp1, edgelets1, threshold_reestimate=5)
    self.vis_model(image,vp1)
    return vp1, edgelets1, image
    
  def second_vp(self,vp1,edgelets1, image):
    edgelets2 = self.remove_inliers(vp1, edgelets1, 10)
    vp2 = self.ransac_vanishing_point(edgelets2, num_ransac_iter=2000,
                            threshold_inlier=5)
    vp2 = self.reestimate_model(vp2, edgelets2, threshold_reestimate=5)
    self.vis_model(image, vp2)
    return vp2,edgelets2

  def third_vp(self, vp1, vp2, image):
    u = vp1.copy()
    v = vp2.copy()
    u_T = np.transpose(u)
    prod = np.matmul(u_T, v)
    f = np.sqrt(prod)
    u[2] = f
    v[2] = f
    w = np.cross(u,v)
    vp3 = [w[0]/w[2], w[1]/w[2],1]
    vp3 = np.array(vp3)
    self.vis_model(image, vp3)
    return vp3, f

  def all_vp(self):
    vp1,edgelets1,image = self.first_vp()
    vp2,_ = self.second_vp(vp1,edgelets1,image)
    vp3,_ = self.third_vp(vp1,vp2,image)
    return vp1, vp2, vp3