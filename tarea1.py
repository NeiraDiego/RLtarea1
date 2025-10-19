import numpy as np
import gym
import gym_gridworld
env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 100 

LEARNING_RATE = 0.2  
GAMMA = 0.90 

epsilon = 1

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

def qlearning(env, epsilon):
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.
    rewards = []
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action = env.action_space.sample() 
            else:
                action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            next_state, reward, done, _ = env.step(action) #Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estado siguiente, recompensa, ¿terminó?).
            rewards_epi=rewards_epi+reward

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]) #Calcula la nueva Q table.

            state = next_state

            if (MAX_STEPS-2)<actual_step:
            	print (f"Episode {episode} rewards: {rewards_epi}")
            	print(f"Value of epsilon: {epsilon}") 
            	if epsilon > 0.1: epsilon -= 0.0001

            if done:
                print (f"Episode {episode} rewards: {rewards_epi}") 
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001
                break  

    print(Q)
    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q


def sarsa(env, epsilon):
    rewards = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles
    
    Q = np.zeros((STATES, ACTIONS))
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        
        if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

        for actual_step in range(MAX_STEPS):

            next_state, reward, done, _ = env.step(action)
            
            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * Q[next_state, action2] - Q[state, action]) #Calcula la nueva Q table.
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2

            if (MAX_STEPS-2)<actual_step:
            	print (f"Episode {episode} rewards: {rewards_epi}")
            	print(f"Value of epsilon: {epsilon}") 
            	if epsilon > 0.1: epsilon -= 0.0001

            if done:
                print (f"Episode {episode} rewards: {rewards_epi}") 
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001
                break   

    print(Q)
    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q


#Función para correr juegos siguiendo una determinada política
def playgames(env, Q, num_games, render = True):
    wins = 0
    env.reset()
    #pause=input()
    env.render()

    for i_episode in range(num_games):
        rewards_epi=0
        observation = env.reset()
        t = 0
        while True:
            action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render: env.render()
            pause=input()
            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    pause=input()
    env.close()
    print("Victorias: ", wins)




Q = sarsa(env, epsilon)
#Q = qlearning(env, epsilon)
playgames(env, Q, 100, True)
env.close()



#_ =env.step(env.action_space.sample())
