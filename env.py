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
        # connect to the AirSim simulator 
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # TODO: Set initial position
        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print(self.state.x_val, self.state.y_val, self.state.z_val)
        self.quad_offset = (0, 0, 0)
        initX = 162
        initY = -320
        initZ = -150

        self.client.takeoffAsync().join()
        print("take off moving positon")
        self.client.moveToPositionAsync(initX, initY, initZ, 5).join()
        #print("moving velocity")
        #self.client.moveByVelocityAsync(1, 0, 0.8, 5).join()

    def step(self, action):
        print("doing step")
        self.quad_offset = self.interpret_action(action)
        print("quad_offset: ", self.quad_offset)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(quad_vel.x_val+self.quad_offset[0], quad_vel.y_val+self.quad_offset[1], quad_vel.z_val+self.quad_offset[2], 0.5).join()
        time.sleep(0.5)
        #self.client.moveToPositionAsync(quad_state.x_val+self.quad_offset[0], quad_state.y_val+self.quad_offset[1], quad_state.z_val+self.quad_offset[2], 5).join()
        #time.sleep(0.5)
        
        collision_info = self.client.simGetCollisionInfo()
        
        if collision_info.has_collided:
            print("Collsion at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
                pprint.pformat(collision_info.position),
                pprint.pformat(collision_info.normal),
                pprint.pformat(collision_info.impact_point),
                collision_info.penetration_depth, collision_info.object_name, collision_info.object_id
            ))
            print(collision_info.object_name)
            print(type(collision_info.object_name))
        #time.sleep(0.5)
        #self.client.hoverAsync().join()
        

        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        print("state x:",quad_state.x_val, " y: ",quad_state.y_val, " z: ",quad_state.z_val)

        reward = self.compute_reward(quad_state, quad_vel, collision_info)
        state = self.get_obs()
        done = self.isDone(reward)
        return state, reward, done
    
    def set_init_pose(self):
        initX = 162
        initY = -320
        initZ = -150
        pose.position.x_val = initX
        pose.position.y_val = initY
        pose.position.z_val = initZ
        self.client.simSetVehiclePose(pose, True)

    def reset(self):
        """set initial state"""
        self.quad_offset = (0, 0, 0)
        initX = 162
        initY = -320
        initZ = -150

        self.client.takeoffAsync().join()
        print("take off moving positon")
        self.client.moveToPositionAsync(initX, initY, initZ, 5).join()

        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        obs = self.transform_input(responses)

        return obs

    def get_obs(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        obs = self.transform_input(responses)
        return obs

    def get_distance(self, quad_state):
        #pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]
        pts = np.array([-10, 10, -10])      
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt-pts)
        print("distance: ", dist)
        return dist

    
    def compute_reward(self, quad_state, quad_vel, collision_info):
        """goal -10, 10, -10"""
        collision_info.has_collided == False
        thresh_dist = 7
        max_dist = 500
        beta = 1

        z = -10

        if collision_info.has_collided:
            if 'Cube' in collision_info.object_name:
                dist = 10000000
                dist = self.get_distance(quad_state)
                #for i in range(0, len(pts)-1):
                #    dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

                print("distance: ", dist)
                if dist > thresh_dist:
                    if dist > max_dist:
                        reward = -100
                    else:
                        reward = 0
                else:
                    reward = 500
            else:
                reward = -100
        """
        else:    
            dist = 10000000
            dist = self.get_distance()
            #for i in range(0, len(pts)-1):
            #    dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

            print("distance: ", dist)
            if dist > thresh_dist:
                if dist > max_dist:
                    reward = -100
                else:
                    reward = 0
            else:
                reward = 500
            #if dist > thresh_dist:
            #    reward = -10
            #else:
            #    reward_dist = (math.exp(-beta*dist) - 0.5) 
            #    reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            #    reward = reward_dist + reward_speed
        print(reward)
        """
        return reward

    def isDone(self, reward):
        done = 0
        if  reward <= -10:
            done = 1
        elif reward > 499:
            done = 1
        return done

    def transform_input(self, responses):
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4) # reshape array to 4 channel image array H X W X 3

        # original image is fliped vertically
        img2d = np.flipud(img_rgba)    

        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 

        return im_final

    def interpret_action(self, action):
        scaling_factor = 5
        if action.item() == 0:
            self.quad_offset = (0, 0, 0)
        elif action.item() == 1:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action.item() == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action.item() == 3:
            self.quad_offset = (0, 0, scaling_factor)
        elif action.item() == 4:
            self.quad_offset = (-scaling_factor, 0, 0)    
        elif action.item() == 5:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action.item() == 6:
            self.quad_offset = (0, 0, -scaling_factor)

        return self.quad_offset
