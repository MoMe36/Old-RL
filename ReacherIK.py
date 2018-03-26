import pygame as pg 
import numpy as np 

def normalize(v): 

	magnitude = np.sqrt(np.sum(v**2) + 1e-5)
	return v/magnitude

class Solver(): 

	def __init__(self, robot, step = 1.):

		self.robot = robot
		self.nb_joints = self.robot.nb_joints
		self.step = step

	def compute_increment(self, direction, with_jaco = False ): 

		V = normalize(direction)#direction/(np.sqrt(np.sum(direction**2)) + 1e-5)		
		V = np.concatenate((V, [0]))
		joints_positions = self.robot.all_joints_positions()		
		
		#Get Jacobian 
		columns = []
		for i in range(self.nb_joints): 
			v = joints_positions[-1] - joints_positions[i]
			c = np.cross(np.array([0,0,1]), v)

			columns.append(c)	

		J = np.array(columns).T
	
		# Compute the delta angles 

		do = normalize(J.T.dot(V))

		if with_jaco: 
			return do*self.step, J
		else: 
			return do*self.step

	def get_only_jacobian(self): 

		joints_positions = self.robot.all_joints_positions()		
		
		#Get Jacobian 
		columns = []
		for i in range(self.nb_joints): 
			v = joints_positions[-1] - joints_positions[i]
			c = np.cross(np.array([0,0,1]), v)

			columns.append(c)	

		J = np.array(columns).T

		return J

	def compute_using_jaco(self, direction, jaco): 

		direction = np.concatenate((direction, [0]))
		V = normalize(direction).reshape(-1,1)#direction/(np.sqrt(np.sum(direction**2)) + 1e-5)		

		# V = np.concatenate((V, [[0]]), axis = 0)
		
		#Get Jacobian 
		# J = jaco.reshape(2,3) 

		do = normalize(jaco.T.dot(V)).reshape(-1)
		return do*self.step


class Robot(): 

	def __init__(self, nb_joints, length_joints, speed = 4, randomize = False): 

		self.nb_joints = nb_joints
		self.uLength = length_joints

		self.speed = speed
		self.angles = np.zeros((nb_joints))+1

		self.solver = Solver(self)

		if randomize: 
			self.angles = np.random.randint(low = -80, high = 80, size =(nb_joints))

		self.compute_positions()

	def rotate(self, action):  ## Old rotate method from actions

		jointIndex = action/2
		direction = 1 if action%2 == 0 else -1

		self.angles[int(jointIndex)] += direction*self.speed

	def rotate_ik(self, delta): # rotate from IK result
		print('delta:{} '.format(delta))
		for i in range(len(self.angles)): 
			print('Angle {} = {}'.format(i,self.angles[i]))
			self.angles[i] += delta[i]
			print('Angle {} = {}'.format(i,self.angles[i]))

	def rotate_jaco(self, direction): 

		increment = self.solver.compute_increment(direction)

		# print(increment)
		for i in range(increment.shape[0]):
			self.angles[i] += increment[i]*self.speed

	def rotate_jaco_and_jaco(self, direction): 

		increment, jaco = self.solver.compute_increment(direction, with_jaco = True)
		for i in range(increment.shape[0]): 
			self.angles[i] += increment[i]*self.speed

		return jaco

	def rotate_from_jaco(self, direction, jaco): 

		increment = self.solver.compute_using_jaco(direction, jaco)
		for i in range(len(self.angles)): #'''increment.shape[0]'''): 
			self.angles[i] += increment[i]*self.speed

		return jaco 

	def compute_positions(self): 
		self.normalize_angles()
		self.points = np.zeros((self.nb_joints+1,2))
		for i in range(self.nb_joints+1): 
			if i == 0: 
				self.points[i,:] = np.array([0.5,0.5])
			else:
				angle = (self.angles[i-1] + 90)%360 
				angle = np.radians(angle)
				self.points[i,:] = self.points[i-1,:] + self.uLength*np.array([np.cos(angle), np.sin(angle)])

	def normalize_angles(self): 
		for i in range(len(self.angles)): 

			self.angles[i] = (self.angles[i])%360

	def joints_positions(self): 
		points = []
		for i,p in enumerate(self.points): 
			if i != 0: 
				points.append(p)	
		return points

	def all_joints_positions(self): 
		points = []
		for i,p in enumerate(self.points): 
			points.append(p)
		return points  

	def draw(self, screen, screenSize): 

		for j in range(self.nb_joints): 
			
			p0 = self.points[j].copy()
			p1 = self.points[j+1].copy()

			p0[0] = (p0[0]*screenSize[0])
			p0[1] = (screenSize[1]*(1-p0[1]))

			p1[0] = (p1[0]*screenSize[0])
			p1[1] = (screenSize[1]*(1-p1[1]))

			pg.draw.line(screen, (220,150,20), p0.astype(int), p1.astype(int), 5)
			pg.draw.circle(screen, (150,250,0), p0.astype(int), 10)
			pg.draw.circle(screen, (150,250,250), p1.astype(int), 10) 

class Target(): 

	def __init__(self, limits = [0.2,0.8,0.2,0.7], radius = 15):

	

		# margin_x = np.random.uniform(low = 0., high = 0.1)
		# margin_y = np.random.uniform(low = 0., high = 0.1)

		# case = np.random.randint(4)
		# if case == 0:
		# 	self.position = np.array([margin_x,margin_y])
		# elif case == 1: 
		# 	self.position = np.array([1-margin_x, margin_y])
		# elif case == 2: 
		# 	self.position = np.array([margin_x, 1-margin_y])
		# else: 
		# 	self.position = np.array([1-margin_x, 1-margin_y])


		x = np.random.uniform(low = limits[0], high = limits[1])
		y = np.random.uniform(low = limits[2], high = limits[3])
		self.position = np.array([x,y])
		

		self.radius = radius


	def draw(self, screen, screenSize): 

		pos = self.position.copy()
		pos[1] = screenSize[1]*(1-pos[1])
		pos[0] *= screenSize[0]
		

		pg.draw.circle(screen, (250,0,0), pos.astype(int), self.radius)
		pg.draw.circle(screen, (250,250,250), pos.astype(int), int(self.radius*2/3))
		pg.draw.circle(screen, (250,0,0), pos.astype(int), int(self.radius/3))

class World():

	def __init__(self, robot_joints = 2, joints_length = 0.18, robot_speed = 3,
				randomize_robot = False, randomize_target = False, reset_robot = False, 
				target_limits = [0.2,0.8,0.2,0.8], max_steps = 200,
				state_type = 'angles',target_description = 'position'):

		self.randomize_robot = randomize_robot
		self.randomize_target = randomize_target
		self.target_limits = target_limits
		self.max_steps = max_steps
		self.robot_speed = robot_speed
		self.reset_robot = reset_robot

		self.robot = Robot(robot_joints,joints_length, speed = robot_speed, randomize = randomize_robot)
		self.target = Target(self.target_limits)

		self.listed_positions = False
		self.robotParameters = [robot_joints, joints_length, robot_speed]

		self.steps = 0

		self.currentDistance = (self.target.position - self.robot.points[-1])**2
		self.currentDistance = np.sqrt(self.currentDistance)

		self.render_ready = False
		self.state_type = state_type
		self.target_description = target_description

		self.possible_actions = {0:np.array([0.,1.]),
								 1:np.array([1.,0.]),
								 2:np.array([0.,-1.]), 
								 3:np.array([-1.,0.])
								 # 4:np.array([1.,1.]),
								 # 5:np.array([-1.,-1.]), 
								 # 6:np.array([1.,-1.]),
								 # 7:np.array([-1.,1.])
								}

	def setTargetPosition(self, pos): 
		#self.target.position = pos
		self.target_positions = pos
		self.target.position = np.array(pos[0])
		self.target_position_iterator = 0
		self.listed_positions = True

	def initRender(self, size = [700,700]):

		pg.init()
		self.screen = pg.display.set_mode(size)
		self.clock = pg.time.Clock()
		self.size = size

	def render(self): 

		if not self.render_ready: 
			self.initRender()
			self.render_ready = True

		time = 30
		self.clock.tick(time)
		self.screen.fill((0,0,0))
		self.draw(self.screen, self.size)

		pg.display.flip() 

	def draw(self,screen, screenSize): 

		self.robot.draw(screen, screenSize)
		self.target.draw(screen, screenSize)

	def observe_ik(self): 

		target_position = self.target.position
		joints_positions = self.robot.all_joints_positions()

		zero = np.zeros((len(joints_positions),1))
		joints_positions = np.concatenate((joints_positions,zero), axis = 1)
		target_position = np.concatenate((target_position, [0]))

		return joints_positions, target_position

	def observe(self): 

		# ------------------------------------------------------------------
		# State is distance between the effector and the ball in the following form -> [dX_Positive, dX_Negative, dY_Pos, dY_Neg]
		# ------------------------------------------------------------------

		targetPosition = self.target.position
		effectorPosition = self.robot.points[-1]

		vector = targetPosition - effectorPosition
		distance = np.sqrt(np.sum(vector**2))
		
		if self.state_type == 'positions': 
			state = []
			pos = self.robot.joints_positions()
			for p in pos: 
				for e in p:
					state.append(e)

		# ------------------------------------------------------------------
		# New state representation using angles
		# ------------------------------------------------------------------
		if self.state_type == 'angles':
			state = []
			for a in self.robot.angles: 
				state.append(np.cos(np.radians(a)))
				state.append(np.sin(np.radians(a)))
		

		
		if self.target_description == 'position':
			for p in targetPosition: 
				state.append(p)
		elif self.target_description == 'vector': 
			for p in vector: 
				state.append(p)
		else: 
			raise KeyError

		# ------------------------------------------------------------------
		# ------- Reward -----------
		# ------------------------------------------------------------------

		if distance > 0.5: reward = -0.1
		else: reward = 0.1*np.exp(-10*distance)



		angle_penalty = 0.7*np.abs(self.robot.angles[0]) + 0.1*np.abs(self.robot.angles[1])

		self.currentDistance = distance

		# ------------------------------------------------------------------
		# ------- Completion -------- 
		# ------------------------------------------------------------------

		complete = False
		success = 0
		i = 0

		if distance < 0.03:  # target reached
			complete = True
			success = 1
			reward = 10.

		if self.steps > self.max_steps: 
			complete = True
			reward = -1.

		return state, reward, complete, success


	def step(self, action): 

		self.robot.rotate(action)
		self.robot.compute_positions()

		self.steps += 1

		return self.observe()

	def step_ik(self, delta): 

		self.robot.rotate_ik(delta)
		self.robot.compute_positions()

		self.steps += 1

		return self.observe()

	def step_jaco(self, direction):

		self.robot.rotate_jaco(self.possible_actions[direction])
		self.robot.compute_positions()
		self.steps += 1
		return self.observe()	

	def step_jaco_real_vector(self, direction): 

		j = self.robot.solver.get_only_jacobian()
		self.robot.rotate_jaco(direction)
		self.robot.compute_positions()
		self.steps += 1
		return [self.observe(), j]

	def step_jaco_and_jaco(self, direction):

		j = self.robot.rotate_jaco_and_jaco(self.possible_actions[direction])
		self.robot.compute_positions()
		self.steps += 1
		return [self.observe(),j]

	def step_from_jacobian_real_vector(self, direction, jaco): 

		self.robot.rotate_from_jaco(direction, jaco)
		self.robot.compute_positions()
		self.steps += 1

		return self.observe()

	def step_from_jacobian_with_action_as_direction(self, direction, jaco): 

		self.robot.rotate_from_jaco(self.possible_actions[direction], jaco)
		self.robot.compute_positions()
		self.steps += 1

		return self.observe()

	def randomAction(self): 

		maxActions = len(self.possible_actions) #self.robot.nb_joints*2
		action = np.random.randint(maxActions)
		return action

	def get_jaco_size(self): 

		return 2*self.robot.nb_joints

	def action_space_size(self): 
		return len(self.possible_actions)

	def observation_space_size(self): 
		s,_,_,_ = self.observe()
		return len(s)

	def get_env_infos(self): 
		return [self.observation_space_size(), self.action_space_size()]

	def reset(self): 

		self.steps = 0
		if self.reset_robot: 
			self.robot = Robot(self.robotParameters[0], 
						   self.robotParameters[1],
						   speed = self.robotParameters[2],
						   randomize= self.randomize_robot)


		if self.randomize_target: 
			self.target = Target(self.target_limits)
		if self.listed_positions: 
			self.target_position_iterator = (self.target_position_iterator+1)%len(self.target_positions)
			self.target.position = np.array(self.target_positions[self.target_position_iterator])


		state,_,__,___ = self.observe()
		return state

	
	def close(self): 
		pg.quit()


# e = World()
# e.reset()
# e.initRender()

# while True: 
# 	e.render()
# 	action = e.randomAction()
# 	e.step(action)