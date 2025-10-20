import numpy as np
import os, csv
import time
import gym
import gym_gridworld
env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000
#a1: para conseguir buen resultado en qlearning map2 aumente steps a 1000
MAX_STEPS = 1000

#a2: para conseguir 2q uso 0.5
LEARNING_RATE = 0.2 #0.2  
#a1 para conseguir buen resultado en sarsa map2 aumente gamma a .99
GAMMA = 0.99 

epsilon = 0.7
lambda_ = 0.7

RESULTS_DIR = "./resultados"

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def save_q_to_csv(Q, filepath):
    # Guarda la Q-table con una fila por estado y columnas por acción
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        # encabezado: A0, A1, ...
        w.writerow([f"A{j}" for j in range(Q.shape[1])])
        for i in range(Q.shape[0]):
            w.writerow(list(Q[i]))

def qlearning(env, epsilon):
    ensure_dir(RESULTS_DIR)
    rewards_csv = os.path.join(RESULTS_DIR, "rewards_por_episodios_map2_qlearning.csv")
    # abrir en modo write para empezar limpio cada corrida
    with open(rewards_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["episode", "rewards_epi", "epsilon"])  # header

        STATES = env.n_states
        ACTIONS = env.n_actions
        Q = np.zeros((STATES, ACTIONS))
        rewards = []

        for episode in range(EPISODES):
            rewards_epi = 0
            state = env.reset()
            for actual_step in range(MAX_STEPS):
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state, :])

                next_state, reward, done, _ = env.step(action)
                rewards_epi += reward

                # update Q
                Q[state, action] = Q[state, action] + LEARNING_RATE * (
                    reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]
                )
                state = next_state

                if (MAX_STEPS - 2) < actual_step:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001

                if done:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    rewards.append(rewards_epi)
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001
                    break

            # si no hubo 'done', igual registramos el episodio
            writer.writerow([episode, rewards_epi, epsilon])

    print(Q)
    # guarda la Q-table
    q_csv = os.path.join(RESULTS_DIR, "q_map2_qlearning.csv")
    save_q_to_csv(Q, q_csv)

    return Q

def double_qlearning(env, epsilon):
    ensure_dir(RESULTS_DIR)

    # CSV de recompensas por episodio
    rewards_csv = os.path.join(RESULTS_DIR, "rewards_por_episodios_map2_double_qlearning.csv")
    with open(rewards_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["episode", "rewards_epi", "epsilon"])  # header

        STATES = env.n_states
        ACTIONS = env.n_actions
        # Dos tablas Q
        Q1 = np.zeros((STATES, ACTIONS))
        Q2 = np.zeros((STATES, ACTIONS))

        for episode in range(EPISODES):
            rewards_epi = 0.0
            state = env.reset()

            for actual_step in range(MAX_STEPS):
                # Política ε-greedy sobre la suma/avg de Q1 y Q2
                Qsum = Q1[state, :] + Q2[state, :]
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Qsum)

                next_state, reward, done, _ = env.step(action)
                rewards_epi += reward

                # --- Actualización Double Q-Learning ---
                if np.random.rand() < 0.5:
                    # Actualiza Q1 usando acción argmax en Q1 y valorado con Q2
                    a_star = np.argmax(Q1[next_state, :])
                    td_target = reward + GAMMA * Q2[next_state, a_star]
                    Q1[state, action] += LEARNING_RATE * (td_target - Q1[state, action])
                else:
                    # Actualiza Q2 usando acción argmax en Q2 y valorado con Q1
                    a_star = np.argmax(Q2[next_state, :])
                    td_target = reward + GAMMA * Q1[next_state, a_star]
                    Q2[state, action] += LEARNING_RATE * (td_target - Q2[state, action])
                # --------------------------------------

                state = next_state

                # Mantengo tu patrón de impresión y decaimiento de epsilon
                if (MAX_STEPS - 2) < actual_step:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001

                if done:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001
                    break

            # Log por episodio (haya 'done' o no)
            writer.writerow([episode, rewards_epi, epsilon])

    # Q combinada para inspección/uso (lo habitual en Double-Q)
    Q_comb = 0.5 * (Q1 + Q2)
    print(Q_comb)

    # Guarda la Q combinada en CSV
    q_csv = os.path.join(RESULTS_DIR, "q_map2_double_qlearning.csv")
    save_q_to_csv(Q_comb, q_csv)

    return Q_comb


def sarsa(env, epsilon):
    ensure_dir(RESULTS_DIR)
    rewards_csv = os.path.join(RESULTS_DIR, "rewards_por_episodios_map2_sarsa.csv")
    with open(rewards_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["episode", "rewards_epi", "epsilon"])  # header

        rewards = []
        STATES = env.n_states
        ACTIONS = env.n_actions
        Q = np.zeros((STATES, ACTIONS))

        for episode in range(EPISODES):
            rewards_epi = 0
            state = env.reset()

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            for actual_step in range(MAX_STEPS):
                next_state, reward, done, _ = env.step(action)

                if np.random.uniform(0, 1) < epsilon:
                    action2 = env.action_space.sample()
                else:
                    action2 = np.argmax(Q[next_state, :])

                # update Q
                Q[state, action] = Q[state, action] + LEARNING_RATE * (
                    reward + GAMMA * Q[next_state, action2] - Q[state, action]
                )

                rewards_epi += reward
                state = next_state
                action = action2

                if (MAX_STEPS - 2) < actual_step:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001

                if done:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    rewards.append(rewards_epi)
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001
                    break

            # registra el episodio (haya 'done' o no)
            writer.writerow([episode, rewards_epi, epsilon])

    print(Q)
    # guarda la Q-table
    q_csv = os.path.join(RESULTS_DIR, "q_map2_sarsa.csv")
    save_q_to_csv(Q, q_csv)

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
            #pause=input()
            time.sleep(0.1)
            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    env.close()
    print("Victorias: ", wins)

def sarsa_lambda(env, epsilon, lambda_):
    ensure_dir(RESULTS_DIR)
    rewards_csv = os.path.join(RESULTS_DIR, "rewards_por_episodios_map1_sarsa_lambda.csv")
    with open(rewards_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["episode", "rewards_epi", "epsilon"])

        STATES, ACTIONS = env.n_states, env.n_actions
        Q = np.zeros((STATES, ACTIONS), dtype=float)

        for episode in range(EPISODES):
            E = np.zeros_like(Q)  # eligibility traces
            rewards_epi = 0.0
            state = env.reset()

            # elegir acción inicial (ε-greedy)
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            for actual_step in range(MAX_STEPS):
                next_state, reward, done, _ = env.step(action)
                rewards_epi += reward

                # acción siguiente (on-policy)
                if not done:
                    if np.random.uniform(0, 1) < epsilon:
                        next_action = env.action_space.sample()
                    else:
                        next_action = np.argmax(Q[next_state, :])

                # TD error
                if done:
                    td_target = reward
                else:
                    td_target = reward + GAMMA * Q[next_state, next_action]
                delta = td_target - Q[state, action]

                # replacing trace
                E[state, action] = 1.0

                # actualización para todas las (s,a)
                Q += LEARNING_RATE * delta * E
                # decaimiento de trazas
                if done:
                    E[:] = 0.0
                else:
                    E *= GAMMA * lambda_

                # mover estado/acción
                state = next_state
                if not done:
                    action = next_action

                # mantener tu impresión/decaimiento de epsilon
                if (MAX_STEPS - 2) < actual_step:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001

                if done:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001
                    break

            writer.writerow([episode, rewards_epi, epsilon])

    print(Q)
    save_q_to_csv(Q, os.path.join(RESULTS_DIR, "q_map1_sarsa_lambda.csv"))
    return Q

def q_lambda(env, epsilon, lambda_):
    ensure_dir(RESULTS_DIR)
    rewards_csv = os.path.join(RESULTS_DIR, "rewards_por_episodios_map1_q_lambda.csv")
    with open(rewards_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["episode", "rewards_epi", "epsilon"])

        STATES, ACTIONS = env.n_states, env.n_actions
        Q = np.zeros((STATES, ACTIONS), dtype=float)

        for episode in range(EPISODES):
            E = np.zeros_like(Q)  # eligibility traces
            rewards_epi = 0.0
            state = env.reset()

            for actual_step in range(MAX_STEPS):
                # acción ε-greedy para comportarse (pero estimamos con greedy)
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state, :])

                next_state, reward, done, _ = env.step(action)
                rewards_epi += reward

                # objetivo off-policy (greedy en next_state)
                if done:
                    td_target = reward
                    greedy_next = None
                else:
                    greedy_next = np.argmax(Q[next_state, :])
                    td_target = reward + GAMMA * Q[next_state, greedy_next]

                delta = td_target - Q[state, action]

                # replacing trace
                E[state, action] = 1.0

                # actualización
                Q += LEARNING_RATE * delta * E

                # Watkins: si la acción que REALMENTE se tomó en next_state no es greedy, E=0
                if done:
                    E[:] = 0.0
                else:
                    # ¿la próxima acción de comportamiento sería greedy?
                    # (miramos lo que haríamos ε-greedy para la siguiente decisión)
                    # Si epsilon > 0, podría no ser greedy; la regla formal mira la acción tomada.
                    # Para aproximar en este esquema paso a paso:
                    # - si la política de comportamiento es greedy (prob. 1-ε) y
                    #   la acción elegida es el argmax => mantenemos E; de lo contrario, anulamos.
                    if np.random.uniform(0, 1) < epsilon:
                        # no-greedy -> anular E
                        E[:] = 0.0
                    else:
                        # se tomaría la greedy: mantener y decaer
                        E *= GAMMA * lambda_

                # mover
                state = next_state

                # mantener tu impresión/decaimiento de epsilon
                if (MAX_STEPS - 2) < actual_step:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001

                if done:
                    print(f"Episode {episode} rewards: {rewards_epi}")
                    print(f"Value of epsilon: {epsilon}")
                    if epsilon > 0.1:
                        epsilon -= 0.0001
                    break

            writer.writerow([episode, rewards_epi, epsilon])

    print(Q)
    save_q_to_csv(Q, os.path.join(RESULTS_DIR, "q_map1_q_lambda.csv"))
    return Q


#Q = sarsa(env, epsilon)
#Q = qlearning(env, epsilon)
#Q = double_qlearning(env, epsilon)
#Q = sarsa_lambda(env, epsilon=epsilon, lambda_=lambda_)
Q = q_lambda(env,     epsilon=epsilon, lambda_=lambda_)
playgames(env, Q, 10, True)
env.close()



#_ =env.step(env.action_space.sample())
