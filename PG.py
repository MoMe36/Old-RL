import torch 
import numpy as np 
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable
import robotWorld as World
import matplotlib.pyplot as plt 


plt.style.use('ggplot')


def discountReward(r, gamma = 0.99): 

	discounted = np.zeros_like(r)
	current = 0

	for i in reversed(range(r.shape[0])): 
		current = current*gamma + r[i]
		discounted[i] = current
	return discounted



# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						     World Creation 
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-



# robotJoints controls the number of joints in the robot 
# world is designed to be inside a [0->1], [0->1] space. Hence, robotJointsLength*robotJoints should be less than 1
# State is distance between the effector and the ball in the following form -> [dX_Positive, dX_Negative, dY_Pos, dY_Neg]
# Actions [rotate right, rotate left] for i in range(nbJoints)
# ex: for 2 joints: [J0RotateRight, J0RotateLeft, J1RotateRight, J1RotateLeft]

env = World.World(robotJoints = 2, robotJointsLength = 0.35, 
	randomizeRobot = False, randomizeTarget = False, 
	groundHeight =0.05, targetLimits = [0.2,0.8,0.1,0.6], maxSteps = 200)

observation_space = 4
action_space = env.actionSpaceSize()

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						     Model Creation 
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


hiddenSize = 100
model = nn.Sequential(nn.Linear(observation_space, hiddenSize), nn.ReLU(), nn.Linear(hiddenSize, action_space), nn.Softmax())
adam = optim.Adam(model.parameters(), 1e-3)

successiveActions = 5 # number of frames before choosing new action given perceptions

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						    Learning Loop
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

epochs = 2000
updateFreq = 5


informationFreq = 100
successes = 0
success_history = []


for epoch in range(epochs): 

	steps = 0
	ep_history = []
	s = env.reset()
	complete = False
	reward = 0
	while not complete: 

		sTensor = Variable(torch.Tensor(s).type(torch.FloatTensor))
		distrib = (model(sTensor.unsqueeze(0)).data.numpy()).reshape(-1)
		choice = np.random.choice(distrib, p = distrib)
		action = np.argmax(choice == distrib)

		ns, r, complete, success = env.step(action)

		ep_history.append([s, action, r, ns])
		s = ns 
		reward += r 
		steps += 1

		if complete: 

			successes += success

			ep_history = np.array(ep_history)
			rH = discountReward(ep_history[:,2]) # reward history 
			sH = np.vstack(ep_history[:,0])	 # states history 
			aH = ep_history[:,1]		# actions history 

			# Computing gradients 

			indexes = np.arange(steps)*action_space
			indexes = (indexes + aH).astype(int)

			indexesTensor = Variable(torch.from_numpy(indexes).type(torch.LongTensor)) 

			qValues = model.forward(Variable(torch.from_numpy(sH).type(torch.FloatTensor))) # computing probabilities

			qValues = qValues.view(1,-1)
			guilty = torch.index_select(qValues,1,indexesTensor) # selecting probabilities according to actions

			rewardTensor = Variable(torch.from_numpy(rH.astype(float)).float()) 
			loss = -torch.mean(torch.matmul(torch.log(guilty),rewardTensor)) # Negative log likelihood weighted by rewards
			loss.backward()

			if epoch%updateFreq == 0: 
				adam.step()
				adam.zero_grad()

			if epoch%informationFreq == 0: 

				text = 'Epoch: {}/{} - Success: {}/{} '.format(epoch,epochs,successes,informationFreq)
				print(text)
				success_history.append(successes)
				successes = 0
				torch.save(model, 'catcherPG.robot')


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						   Visualisation
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


def mplot(success): 

	success = np.array(success)
	plt.bar(range(success.shape[0]), success)
	plt.title('Successes over slices of {} trials'.format(informationFreq))
	plt.xlabel('Trials')
	plt.ylabel('Successes')
	plt.ylim(0,101)
	plt.show()

mplot(success_history)

env.initRender()
s = env.reset()
steps = 0
epsilon = 0.1
while True: 

	if steps%successiveActions == 0: 

		sTensor = Variable(torch.Tensor(s).type(torch.FloatTensor))
		distrib = (model(sTensor.unsqueeze(0)).data.numpy()).reshape(-1)
		choice = np.random.choice(distrib, p = distrib)
		action = np.argmax(choice == distrib)

	ns,r,d,_ = env.step(action)
	env.render()

	s = ns
	steps += 1 
	if d: 
		steps = 0
		s = env.reset()

			



