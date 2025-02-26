from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math
import time

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
            - Etat sécuritaire
        '''

        t = time.time()
        player = current_state.get_next_player()
        opponent = current_state.compute_next_player()
        depth_limit = 4

        def state_score(state):
            return state.get_player_score(player)-state.get_player_score(opponent)
        
        def state_heuritic(state):
            pieces_left = state.compute_players_pieces_left(player.get_id())
            score = state_score(state)

        def is_worth(state, score, step):
            new_score = state_score(state)
            if new_score == score - 5:
                return False
            return True
            
        def max_value(state, alpha, beta, depth):

            depth += 1
            score = state_score(state)
            step = state.get_step()

            if state.is_done() or depth>=depth_limit:
                return((score, None))

            v_star = - math.inf
            m_star = None

            
            for action in state.get_possible_light_actions():
                new_state = state.apply_action(action)
                
                if step == 0:
                    return (state_score(new_state),action)

                if is_worth(new_state, score, step):
                    
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
            score = state_score(state)

            if state.is_done() or depth>=depth_limit:
                return((score, None))

            v_star = math.inf
            m_star = None

            for action in state.get_possible_light_actions():
                new_state = state.apply_action(action)
                    
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
        dt = time.time() - t
        print(dt)
        return m
