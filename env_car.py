import setup_path 
import airsim

from argparse import ArgumentParser

import numpy as np
import time
import math
import pprint

import csv

class DroneEnv():
    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        car_controls = airsim.CarControls()


    def step(self, action):
        action = agent.act(current_state)
        car_controls = interpret_action(action)
        self.client.setCarControls(car_controls)

        car_state = self.client.getCarState()
        reward = compute_reward(car_state) 
        done = isDone(car_state, car_controls, reward)

        return car_state, reward, done


    def reset(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        car_controls = airsim.CarControls()

        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        obs = self.transform_input(responses)

        return obs

    def get_obs(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        obs = self.transform_input(responses)
        return obs


    def compute_reward(self, car_state):
        MAX_SPEED = 300
        MIN_SPEED = 10
        thresh_dist = 3.5
        beta = 3

        z = 0
        pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
        pd = car_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

        dist = 10000000
        for i in range(0, len(pts)-1):
            dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

        #print(dist)
        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = (math.exp(-beta*dist) - 0.5)
            reward_speed = (((car_state.speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
            reward = reward_dist + reward_speed

        return reward


    def isDone(self, scar_state, car_controls, reward):
        done = 0
        if reward < -1:
            done = 1
            self.client.reset()
            self.client.enableApiControl(False)
        if car_controls.brake == 0:
            if car_state.speed <= 5:
                done = 1
                self.client.reset()
                self.client.enableApiControl(False)
        return done


    def transform_input(self, responses):
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4) # reshape array to 4 channel image array H X W X 3
        print("height, width: ", response.height, response.width)
        # original image is fliped vertically
        img2d = np.flipud(img_rgba)    

        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 

        return im_final

    def interpret_action(self, action):
        car_controls.brake = 0
        car_controls.throttle = 1
        if action == 0:
            car_controls.throttle = 0
            car_controls.brake = 1
        elif action == 1:
            car_controls.steering = 0
        elif action == 2:
            car_controls.steering = 0.5
        elif action == 3:
            car_controls.steering = -0.5
        elif action == 4:
            car_controls.steering = 0.25
        else:
            car_controls.steering = -0.25
        return car_controls
