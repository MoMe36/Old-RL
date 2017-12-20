import numpy as np 
import pygame as pg 


class World():

	def __init__(self, robotJoints = 4, robotJointsLength = 0.18, 
				randomizeRobot = False, randomizeTarget = False, 
				groundHeight = 0.05, targetLimits = [0.2,0.8,0.2,0.8], maxSteps = 200):

		self.randomizeRobot = randomizeRobot
		self.randomizeTarget = randomizeTarget
		self.baseHeight = groundHeight
		self.targetLimits = targetLimits
		self.maxSteps = maxSteps

		self.ground = Ground(groundHeight)
		self.robot = Robot(robotJoints,robotJointsLength, baseHeight = groundHeight, randomize = randomizeRobot)
		self.target = Target(self.targetLimits)

		self.robotParameters = [robotJoints, robotJointsLength]

		self.steps = 0

	def initRender(self, size = [700,700]):

		pg.init()
		self.screen = pg.display.set_mode(size)
		self.clock = pg.time.Clock()
		self.size = size

	def render(self): 

		time = 30
		self.clock.tick(time)
		self.screen.fill((0,0,0))
		self.draw(self.screen, self.size)

		pg.display.flip() 

	def draw(self,screen, screenSize): 

		self.ground.draw(screen, screenSize)
		self.robot.draw(screen, screenSize)
		self.target.draw(screen, screenSize)

	def observe(self): 

		# ------------------------------------------------------------------
		# State is distance between the effector and the ball in the following form -> [dX_Positive, dX_Negative, dY_Pos, dY_Neg]
		# ------------------------------------------------------------------

		targetPosition = self.target.position
		effectorPosition = self.robot.points[-1]

		#print('effectorPosition: {} '.format(effectorPosition))
		#print('targetPosition: {} '.format(targetPosition))

		vector = targetPosition - effectorPosition
		distance = np.sqrt(np.sum(vector**2))

		#print('distance: {} '.format(distance))
		
		state = []
		for element in vector: 
			if element > 0: 
				state.append(element)
				state.append(0)
			else: 
				state.append(0)
				state.append(-element)

		# ------------------------------------------------------------------
		# ------- Reward -----------
		# ------------------------------------------------------------------
		reward = -1


		# ------------------------------------------------------------------
		# ------- Completion -------- 
		# ------------------------------------------------------------------

		complete = False
		success = 0
		i = 0
		for p in self.robot.points:    # checking whether touching the ground
			if p[1] < self.ground.height: 
				complete = True
				reward = 0

		if distance < 0.03:  # target reached
			complete = True
			success = 1
			reward = 10

		if self.steps > self.maxSteps: 
			complete = True
			reward = -1

		return state, reward, complete, success


	def step(self, action): 

		self.robot.rotate(action)
		self.robot.computePositions()

		self.steps += 1

		return self.observe()
		 

	def randomAction(self): 

		maxActions = self.robot.nbJoints*2
		action = np.random.randint(maxActions)
		return action

	def actionSpaceSize(self): 
		return self.robot.nbJoints*2

	def reset(self): 

		self.steps = 0
		self.robot = Robot(self.robotParameters[0], self.robotParameters[1], baseHeight = self.baseHeight, randomize= self.randomizeRobot)
		if self.randomizeTarget: 
			self.target = Target(self.targetLimits)

		state,_,__,___ = self.observe()
		return state



class Target(): 

	def __init__(self, limits = [0.2,0.8,0.2,0.7], radius = 15):

		x = np.random.uniform(low = limits[0], high = limits[1])
		y = np.random.uniform(low = limits[2], high = limits[3])

		self.position = np.array([x,y])
		self.radius = radius


	def draw(self, screen, screenSize): 

		pos = self.position.copy()
		pos[1] = screenSize[1]*(1-pos[1])
		pos[0] *= screenSize[0]
		

		pg.draw.circle(screen, (250,0,0), pos.astype(int), self.radius)
		pg.draw.circle(screen, (250,250,250), pos.astype(int), self.radius*2/3)
		pg.draw.circle(screen, (250,0,0), pos.astype(int), self.radius/3)


class Ground(): 

	def __init__(self, height): 

		self.height = height
		self.color = (150,150,150)

	def draw(self, screen, screenSize, nbL = 10): 

		heightOnScreen = screenSize[1]*(1.-self.height)
		pg.draw.line(screen,self.color, [0, heightOnScreen], [screenSize[0], heightOnScreen])

		inc = screenSize[0]/nbL
		for i in range(nbL): 
			pg.draw.line(screen,self.color, [i*inc, heightOnScreen], [(i+1)*inc, screenSize[1]])
			pg.draw.line(screen,self.color, [i*inc, screenSize[1]], [(i+1)*inc, heightOnScreen])

class Robot(): 

	def __init__(self, nbJoints, LengthJoints, baseHeight = 0.0501, speed = 2, randomize = False): 

		self.nbJoints = nbJoints
		self.uLength = LengthJoints

		self.baseHeight = baseHeight
		self.speed = speed

		self.angles = np.zeros((nbJoints))

		if randomize: 
			self.angles = np.random.randint(low = -80, high = 80, size =(nbJoints))

		self.computePositions()

	def rotate(self, action): 

		jointIndex = action/2
		direction = 1 if action%2 == 0 else -1

		self.angles[jointIndex] += direction*self.speed


	def computePositions(self): 

		self.points = np.zeros((self.nbJoints+1,2))
		for i in range(self.nbJoints+1): 
			if i == 0: 
				self.points[i,:] = np.array([0.5,self.baseHeight])
			else:
				angle = (self.angles[i-1] + 90)%360 
				angle = np.radians(angle)
				self.points[i,:] = self.points[i-1,:] + self.uLength*np.array([np.cos(angle), np.sin(angle)])


	def draw(self, screen, screenSize): 

		for j in range(self.nbJoints): 
			
			p0 = self.points[j].copy()
			p1 = self.points[j+1].copy()

			p0[0] = (p0[0]*screenSize[0])
			p0[1] = (screenSize[1]*(1-p0[1]))

			p1[0] = (p1[0]*screenSize[0])
			p1[1] = (screenSize[1]*(1-p1[1]))

			pg.draw.line(screen, (220,150,20), p0.astype(int), p1.astype(int), 5)
			pg.draw.circle(screen, (150,250,250), p0.astype(int), 10)
			pg.draw.circle(screen, (150,250,250), p1.astype(int), 10)



# w = World(robotJoints = 2, robotJointsLength = 0.35, 
# 	randomizeRobot = False, randomizeTarget = False, 
# 	groundHeight =0.05, targetLimits = [0.2,0.8,0.1,0.6])

# w.initRender()

# while True:
# 	state, reward, complete, success = w.step(w.randomAction())
	
# 	if complete: 
# 		w.reset()
# 		raw_input()

# 	w.render()

