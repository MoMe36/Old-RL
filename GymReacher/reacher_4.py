import numpy as np 
import pygame as pg 
import arcade 

import gym 
from gym import error, spaces, utils 
from gym.utils import seeding 

def compute_effector_position(config, joints_length): 

	j_current = np.array([0.5,0.5])
	config = config.reshape(1,-1)
	for i in range(config.shape[1]): 
		c = config[:,i].reshape(-1)
		angle_mat = np.hstack([np.cos(c), np.sin(c)])
		j_current = j_current +joints_length* angle_mat

	return j_current

def distance(p1, p2): 
	return np.sqrt(np.sum(np.power(p1-p2,2)))

def normalize_angle(angle): 
	return angle%(np.pi*2.)

class Target: 

	def __init__(self, max_distance, pos_ini = None, radius = 20): 

		angle = np.random.uniform(0., np.pi*2.)
		ratio = np.random.uniform(0.1,0.5)
		pos = np.array([0.5 + np.cos(angle)*ratio*max_distance, 0.5 + np.sin(angle)*ratio*max_distance])

		self.pos = pos_ini if pos_ini != None else pos
		self.radius = radius

	@property
	def draw_info(self):
		return self.pos.copy(), self.radius


class Robot: 

	def __init__(self, nb_joints, joints_length, config_ini = None): 

		self.nb_joints = nb_joints-1 
		self.joints_length = joints_length 
		self.config = np.random.uniform(0.,np.pi*2, (self.nb_joints)) # last joint doesn't rotate
		
	def set_config(self, config): 

		self.config = config.copy()

	def rotate(self, vec): 
		config = self.config.copy()
		# print(config)
		config += vec 
		# print(config)
		# input()
		config = config%(np.pi*2)
		self.config = config.copy()

	@property
	def draw_info(self): 

		positions = np.zeros((self.nb_joints+1, 2))
		positions[0] = [0.5,0.5]

		for i in range(1,positions.shape[0]):

			x = positions[i-1,0] + self.joints_length*np.cos(self.config[i-1])
			y = positions[i-1,1] + self.joints_length*np.sin(self.config[i-1])

			positions[i,:] = [x,y]

		lines = np.zeros((self.nb_joints, 4))
		for i in range(lines.shape[0]): 
			x = positions[i:i+2,:]
			lines[i,:] = positions[i:i+2,:].reshape(-1)
		return positions, lines

	@property
	def angles(self): 
		return self.config.copy()

class World(gym.Env): 

	metadata = {'render.modes':['human']}

	def __init__(self, nb_joints = 4, joints_length = 0.2, max_steps = 250):

		
		self.nb_joints, self.joints_length = nb_joints, joints_length
		self.robot = Robot(nb_joints, joints_length)
		self.max_distance = (self.nb_joints)*self.joints_length
		self.target = Target(self.max_distance)

		self.steps = 0
		self.max_steps = max_steps

		self.render_ready = False 
		self.render_size = [700, 700]


		low = np.zeros(5)
		high = np.array([np.pi*2 for i in range(3)] + [3.,3.])

		self.action_space  = spaces.Box(np.ones((3))*(-0.5), np.ones((3))*0.5, dtype = np.float)
		self.observation_space = spaces.Box(low = low, high = high, dtype = np.float)

	def seed(self, seed = None): 

		return

	def step(self, action): 

		self.steps += 1
		self.robot.rotate(action)

		return self.observe()

	def observe(self): 

		angles = self.robot.angles
		vector = self.vector_target_to_end_effector()

		d = self.compute_distance_end_effector_target()
		if d <= 0.03: 
			reward = 1.
			over = True 
		else: 
			reward = -d
			over = False 

		if self.steps > self.max_steps: 
			over = True

		return list(angles.reshape(-1)) + list(vector.reshape(-1)), reward, over, None 

	def reset(self): 

		self.steps = 0 
		self.target = Target(self.max_distance)

		return self.observe()[0]

	def init_render(self): 

		pg.init()
		self.screen = pg.display.set_mode(self.render_size)
		self.clock = pg.time.Clock()

	def render(self, time = 30., mode ='human', close = 'false'): 

		if not self.render_ready: 
			self.init_render()
			self.render_ready = True

		self.clock.tick(time)
		self.screen.fill((0,0,0))
		self.draw()
		pg.display.flip()

	def draw(self): 

		# draw target 

		target_pos, target_radius = self.target.draw_info
		normalized_pos = (target_pos*self.render_size).astype(int)

		pg.draw.circle(self.screen, (250,0,0), normalized_pos, int(target_radius))
		pg.draw.circle(self.screen, (250,250,250), normalized_pos, int(target_radius*2./3))
		pg.draw.circle(self.screen, (250,0,0), normalized_pos, int(target_radius/3.))

		# draw robot

		joint_pos, lines = self.robot.draw_info

		for l in lines: 
			p1 = (l[0:2]*self.render_size).astype(int)
			p2 = (l[2:]*self.render_size).astype(int)
			pg.draw.line(self.screen, (215, 101, 0), p1, p2, int(target_radius/2.))
	
		for jp in joint_pos: 
			normalized_pos =  (self.render_size*jp).astype(int)
			pg.draw.circle(self.screen, (19, 131, 235),normalized_pos, int(target_radius/2.))

	def compute_distance_end_effector_target(self): 

		return distance(self.position_end_effector_target(), self.target.pos)

	def position_end_effector_target(self): 

		return compute_effector_position(self.robot.config, self.joints_length)

	def vector_target_to_end_effector(self): 

		return self.target.pos - self.position_end_effector_target() 

class DebugRender(arcade.Window): 

	def __init__(self, world, size = 700): 

		arcade.Window.__init__(self, size, size, "Gen")
		self.size = size
		self.world = world 

	def on_draw(self): 	

		arcade.start_render()

		pos,lines = self.world.robot.draw_info 
		config = self.world.robot.angles

		for l in lines: 
			scaled_l = l.copy()*self.size
			arcade.draw_line(*scaled_l, (215, 101, 0), 10)
		for i,p in enumerate(pos): 
			scaled_p = p.copy()*self.size
			arcade.draw_circle_filled(scaled_p[0], scaled_p[1], 10, (19, 131, 235))
			if(i < config.shape[0]): 
				arcade.draw_text("{}".format(config[i]), scaled_p[0], scaled_p[1], arcade.color.WHITE, 12)

		target_pos, _ = self.world.target.draw_info
		target_pos *= self.size
		arcade.draw_circle_filled(target_pos[0], target_pos[1], 15, (250, 15, 0))
		arcade.draw_circle_filled(target_pos[0], target_pos[1], 10, (250, 250, 250))
		arcade.draw_circle_filled(target_pos[0], target_pos[1], 5, (250, 15, 0))


		p = compute_effector_position(self.world.robot.config.reshape(1,-1), self.world.robot.joints_length)
		arcade.draw_text("Effector pos: {}\nTarget pos: {}".format(p, distance(p, self.world.target.pos)), p[0]*self.size, p[1]*self.size, arcade.color.WHITE, 12)

		_, r, _, _ = self.world.observe()
		arcade.draw_text("Reward: {}".format(r), 550,650, arcade.color.WHITE, 12)
	def update(self, value): 

		self.world.step(np.random.uniform(-0.1,0.1, (3)))


w = World()
# render = DebugRender(w)
# arcade.run()
# for a in range(1000): 

# 	w.render()
# 	l  = w.step(np.random.uniform(-0.1,0.1, (3)))
# 	print(len(l[0]))