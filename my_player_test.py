from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math
import numpy as np

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        # Add any information you want to store about the player here
        # self.json_additional_info = {}

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        #TODO
        
        '''
            Idées à implémenter :
            - Elagage plus large pour pouvoir augmenter la depth_limit
            - Courbe d'évolution du nombre de cités en fonction du step
            - Etats sécuritaires
            - Mémorisation des états
        '''

        #Getting players and their id
        player = current_state.get_next_player()
        opponent = current_state.compute_next_player()
        player_id = player.get_id()
        opponent_id = opponent.get_id()

        #Initializing the depth limit for the minimax algorithm
        depth_limit = 5

        alpha_player = 0.1
        alpha_opponent = 0.1
        beta_player = 0.1
        beta_opponent = 0.1
        divercity_weight = 0.1
        city_weight = 0.5

        def state_score(state):
            """
            Computes the score gap between the players for the given state.

            Args:
                state (GameState): The current game state.

            Returns:
                int: The score gap between the player and their opponent.
            """
            return state.get_player_score(player)-state.get_player_score(opponent)
        
        def good_start(action):
            """
            Determines whether the given action is a good choice for starting a game.

            Args:
                action (LightAction): The action considered for the first move of the game.

            Returns:
                bool: True if the action is considered good, False otherwise.
            """
            center_pos = [(3,4), (4,3), (4,5), (5,4)]
            if action.data['piece'][1] == 'R':
                return False
            position = action.data['position']
            if position in center_pos:
                return True
            return False
        
        def divercity_estimator(state, position):
            neighbours = state.get_neighbours(position[0],position[1])
            colors = []
            for direction in neighbours:
                if neighbours[direction][0]!='EMPTY':
                    color = neighbours[direction][0].get_type()[0]
                    if color in colors:
                        return 0
                    colors.append(color)
            return 1+len(colors)**2
        
        def state_heuristic(state, step):

            score = state_score(state)
            heuristic = score

            board = state.get_rep().get_env()
            players_pieces_left = state.players_pieces_left

            players_indicators = {player_id : {"cities" : 0, "diversity_score" : 0}, 
                                  opponent_id : {"cities" : 0, "diversity_score" : 0} }
            
            for id in players_pieces_left:
                numbers_pieces_left = list(players_pieces_left[id].values())
                numbers_resources_left = numbers_pieces_left[::2]
                numbers_cities_left = numbers_pieces_left[1::2]
                if id == player_id:
                    heuristic = heuristic - alpha_player * np.var(numbers_resources_left) - beta_player * np.var(numbers_cities_left)
                else:
                    heuristic = heuristic + alpha_opponent * np.var(numbers_resources_left) + beta_opponent * np.var(numbers_cities_left)

            for position in board:
                type = board[position].get_type() #type = "YRW" for YellowResourceWhite
                id = board[position].get_owner_id()
                if type[1] == 'C':
                    players_indicators[id]["diversity_score"] += divercity_estimator(state, position)
                    players_indicators[id]["cities"]+=1

            heuristic = heuristic + (players_indicators[player_id]["diversity_score"]-players_indicators[opponent_id]["diversity_score"])*divercity_weight*(1-step/40)
            heuristic = heuristic + (players_indicators[player_id]["cities"]-players_indicators[opponent_id]["cities"])*city_weight*(1-step/40)

            return heuristic

        def worth(state, heuristic, no_actions_chosen, step):
            if no_actions_chosen:
                return True
            else:
                new_heuristic = state_heuristic(state, step+1)
                if new_heuristic < heuristic:
                    return False
            return True
        
        def worth_opponent(state, heuristic, no_actions_chosen, step):
            if no_actions_chosen:
                return True
            else:
                new_heuristic = state_heuristic(state, step+1)
                if new_heuristic > heuristic:
                    return False
            return True
            
        def max_value(state, alpha, beta, depth):

            depth += 1

            step = state.get_step()
            score = state_score(state)
            heuristic = state_heuristic(state, step)
            no_action_chosen = True

            if state.is_done():
                return((score, None))
            
            if depth>=depth_limit:
                return ((heuristic, None))

            v_star = - math.inf
            m_star = None
            
            for action in state.get_possible_light_actions():
                new_state = state.apply_action(action)
                
                if step == 0:
                    if good_start(action):
                        return (state_score(new_state),action)
                    else:
                        continue

                if worth(new_state, heuristic, no_action_chosen, step):

                    no_action_chosen = False
                    
                    (v,m) = min_value(new_state, alpha, beta, depth)
                    if v > v_star:
                        v_star = v
                        m_star = action
                        alpha = max(alpha, v_star)
                    
                    if v_star >= beta:
                        return (v_star, m_star)
                    
            return (v_star, m_star)
        
        def min_value(state, alpha, beta, depth):

            depth +=1

            step = state.get_step()
            score = state_score(state)
            heuristic = state_heuristic(state, step)
            no_action_chosen = True

            if state.is_done():
                return((score, None))
            
            if depth>=depth_limit:
                return ((heuristic, None))

            v_star = math.inf
            m_star = None

            for action in state.get_possible_light_actions():

                new_state = state.apply_action(action)

                if worth_opponent(new_state, heuristic, no_action_chosen, step):

                    no_action_chosen = False
                    
                    (v,m) = max_value(new_state, alpha, beta, depth)
                    if v < v_star:
                        v_star = v
                        m_star = action
                        beta = min(beta, v_star)

                    if v_star <= alpha:
                        return (v_star, m_star)
                    
            return (v_star, m_star) 
    
        depth = 0
        (v, m) = max_value(current_state, - math.inf, math.inf, depth)
        return m
