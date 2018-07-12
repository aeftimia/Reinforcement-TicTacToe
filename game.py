from numpy.random import choice

class tictactoe:
    # num_states = 5478

    def __init__(self):
        self.moves = {'X': set(), 'O': set()}
        self.players = sorted(self.moves.keys())
        full_board = set()
        for i in range(3):
            for j in range(3):
                full_board.add((i, j))
        self.full_board = full_board
        self.legal_moves = full_board.copy()
        self.state = self.get_state()
        
    def get_state(self):
        return hash(tuple(frozenset(self.moves[player]) for player in self.players))

    def get_reward(self, player):
        for test_player, plays in self.moves.items():
            horizontal_counter = [0] * 3
            vertical_counter = [0] * 3
            diagonal_counter_1 = 0
            diagonal_counter_2 = 0
            for play in plays:
                i, j = play
                horizontal_counter[i] += 1
                vertical_counter[j] += 1
                if i == j:
                    diagonal_counter_1 += 1
                if i == 2 - j:
                    diagonal_counter_2 += 1
            N = frozenset(horizontal_counter) | frozenset(vertical_counter) | frozenset([diagonal_counter_1, diagonal_counter_2])
            if 3 in N:
                if test_player == player:
                    return 1
                else:
                    return -1
        return -0.5

    def is_done(self):
        if not self.legal_moves:
            return True
        if self.get_winner() is not None:
            return True

    def get_winner(self):
        for player in self.players:
            if self.get_reward(player) == 1:
                return player
        return None

    def action(self, move, player):
        self.moves[player].add(move)
        self.legal_moves.remove(move)
        self.state = self.get_state()

    def print_board(self):
        for i in range(3):
            line = []
            for j in range(3):
                for player, moves in self.moves.items():
                    if (i, j) in moves:
                        line.append(player)
                        break
                if (i, j) not in moves:
                    line.append('-')
            print('|'.join(line))
        print('')

class agent:

    def __init__(self, discount_factor, learning_rate, token, environment, weights={}):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.token = token
        self.environment = environment
        self.state = environment.state
        self.weights = weights

    def get_q(self):
        if self.environment.state not in self.weights:
            weights = {}
            for move in self.environment.legal_moves:
                weights[move] = 0
            self.weights[self.environment.state] = weights
        return self.weights

    def update_q(self):
        weights = self.get_q()
        q = self.weights[self.state][self.action]
        q *= (1 - self.learning_rate)
        possible_rewards = weights[self.environment.state].values()
        if not possible_rewards:
            max_expected_reward = 0
        else:
            max_expected_reward = max(possible_rewards)
        reward = self.environment.get_reward(self.token)
        q += self.learning_rate * (reward + self.discount_factor * max_expected_reward)
        self.weights[self.state][self.action] = q

        return q

    def select_action(self):
        q_table = self.get_q()[self.state]

        best_q = None
        for action, q in q_table.items():
            if best_q == None or q > best_q: 
                best_action = action
                best_q = q
        return best_action

    def act(self):
        self.state = self.environment.state
        self.action = self.select_action()
        self.environment.action(self.action, self.token)

board = tictactoe()
discount_factor = 0.25
learning_rate = 1

agent_1 = agent(discount_factor, learning_rate, 'O', board)
agent_2 = agent(discount_factor, learning_rate, 'X', board)
agents = (agent_1, agent_2)

def train(agents, board):
    num_agents = len(agents)
    flag = False
    while not board.is_done():
        for n, agent in enumerate(agents):
            agent.act()
            if flag:
                agents[(n - (num_agents - 1)) % num_agents].update_q()
            if board.is_done():
                update = True
                break
        for i in range(num_agents - 1):
            agents[(n + 1 + i - (num_agents - 1)) % num_agents].update_q()
        flag = True

    # print final position
    board.print_board()
    board.__init__()

N = 2000
for i in range(N):
    train(agents, board)

def get_action(board):
    board.print_board()
    i = None
    j = None
    while (i, j) not in board.legal_moves:
        i = None
        j = None
        while i not in ('0', '1', '2'):
            i = input('row ')
        i = int(i)
        while j not in ('0', '1', '2'):
            j = input('column ')
        j = int(j)
    return (i, j)

agent_2.select_action = lambda: get_action(board)
agents = (agent_1, agent_2)
while True:
    train(agents, board)
    board.print_board()
