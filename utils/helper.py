import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Turn on interactive mode initially

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    plt.figure()  # Create a new figure for each update
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.ylim(ymin=0)
    plt.show()

def turn_off_interactive_mode():
    plt.ioff()

def run_game(agent, game, num_games=1000):
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0

    for _ in range(num_games):
        # get old state
        old_state = agent.get_state(game)

        # get action
        action = agent.get_action(old_state)

        # perform move and get new move
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, action, reward, new_state, done)

        # remember
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            # train long memory
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print('Game', agent.n_game, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_score.append(mean_score)

            plot(plot_scores, plot_mean_score)

    turn_off_interactive_mode()  # Turn off interactive mode after the game loop
    plot(plot_scores, plot_mean_score)  # Display the final graph
