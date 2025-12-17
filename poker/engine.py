"""Texas Hold'em poker engine."""

import random
from typing import Any

from pydantic import BaseModel, Field

from poker.actions import ActionType, BettingRound, Card, PlayerAction
from poker.hand_eval import evaluate_hand, get_hand_name


class Player(BaseModel):
    """Represents a player in the game."""

    player_id: str
    chips: int
    hole_cards: list[Card] = Field(default_factory=list)
    current_bet: int = 0
    total_bet_this_hand: int = 0  # Track total contribution for side pots
    folded: bool = False
    all_in: bool = False
    has_acted_this_round: bool = False
    rebuys: int = 0  # Track rebuys for profit calculation

    def reset_for_new_hand(self) -> None:
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.current_bet = 0
        self.total_bet_this_hand = 0
        self.folded = False
        self.all_in = False
        self.has_acted_this_round = False

    def reset_for_new_round(self) -> None:
        """Reset player state for a new betting round."""
        self.current_bet = 0
        self.has_acted_this_round = False


class ShowdownResult(BaseModel):
    """Result of a hand that went to showdown."""

    winners: list[str]  # player_ids
    pot: int
    revealed_hands: dict[str, tuple[list[Card], str]]  # player_id -> (hole_cards, hand_name)
    board: list[Card]
    hand_number: int
    winnings: dict[str, int]  # player_id -> amount won


class Pot(BaseModel):
    """Represents a pot (main or side pot)."""

    amount: int = 0
    eligible_players: list[str] = Field(default_factory=list)


class OpponentInfo(BaseModel):
    """Public information about an opponent."""

    player_id: str
    chips: int
    current_bet: int
    folded: bool
    all_in: bool
    position: str  # "BTN", "SB", "BB", "UTG", "MP", "CO"
    seat_number: int  # Seats clockwise from player (1 = immediate left)


class GameState(BaseModel):
    """Snapshot of game state for a player."""

    hand_number: int
    betting_round: BettingRound
    pot: int
    community_cards: list[Card]

    # Player's private info
    hole_cards: list[Card]
    player_id: str
    player_chips: int
    player_current_bet: int

    # Multi-opponent info (public)
    opponents: list[OpponentInfo]
    num_active_players: int  # Players who haven't folded
    total_players: int  # Total players at the table

    # Betting info
    current_bet: int  # Current bet to call
    min_raise: int
    call_amount: int  # Amount needed to call

    # Position info
    is_dealer: bool
    is_small_blind: bool
    is_big_blind: bool
    position_name: str  # "BTN", "SB", "BB", "UTG", "MP", "CO"

    # Action history this hand
    action_history: list[str]

    # Blinds
    small_blind: int
    big_blind: int

    # Backwards compatibility properties for heads-up
    @property
    def opponent_id(self) -> str:
        """Return first opponent (for heads-up compatibility)."""
        return self.opponents[0].player_id if self.opponents else ""

    @property
    def opponent_chips(self) -> int:
        """Return first opponent's chips (for heads-up compatibility)."""
        return self.opponents[0].chips if self.opponents else 0

    @property
    def opponent_current_bet(self) -> int:
        """Return first opponent's current bet (for heads-up compatibility)."""
        return self.opponents[0].current_bet if self.opponents else 0

    @property
    def opponent_folded(self) -> bool:
        """Return first opponent's folded status (for heads-up compatibility)."""
        return self.opponents[0].folded if self.opponents else False

    @property
    def opponent_all_in(self) -> bool:
        """Return first opponent's all-in status (for heads-up compatibility)."""
        return self.opponents[0].all_in if self.opponents else False


class Deck:
    """Standard 52-card deck."""

    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    SUITS = ["h", "d", "c", "s"]

    def __init__(self) -> None:
        self.cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        """Reset and shuffle the deck."""
        self.cards = [Card(rank=r, suit=s) for r in self.RANKS for s in self.SUITS]
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[Card]:
        """Deal n cards from the deck."""
        if n > len(self.cards):
            raise ValueError(f"Not enough cards in deck. Requested {n}, have {len(self.cards)}")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt


class PokerEngine:
    """Texas Hold'em poker engine for 2-6 players."""

    def __init__(
        self,
        player_ids: list[str],
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
    ):
        if len(player_ids) < 2 or len(player_ids) > 6:
            raise ValueError("This engine supports 2-6 players")

        self.player_ids = player_ids
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack

        # Initialize players
        self.players: dict[str, Player] = {
            pid: Player(player_id=pid, chips=starting_stack) for pid in player_ids
        }

        # Game state
        self.deck = Deck()
        self.community_cards: list[Card] = []
        self.pot = 0
        self.betting_round = BettingRound.PREFLOP
        self.current_bet = 0  # Current bet level
        self.last_raise_amount = big_blind  # Minimum raise is last raise amount
        self.hand_number = 0

        # Track dealer position (button)
        self.dealer_index = 0  # Index into player_ids

        # Track whose turn it is
        self.current_player_index = 0

        # Action history for this hand
        self.action_history: list[str] = []

        # Hand complete flag
        self._hand_complete = False
        self._showdown_result: ShowdownResult | None = None

    @property
    def num_players(self) -> int:
        """Number of players in the game."""
        return len(self.player_ids)

    @property
    def is_heads_up(self) -> bool:
        """Whether this is a heads-up (2-player) game."""
        return self.num_players == 2

    def remove_eliminated_players(self) -> list[str]:
        """Remove players with 0 chips. Returns list of eliminated player IDs."""
        eliminated = [pid for pid in self.player_ids if self.players[pid].chips <= 0]
        for pid in eliminated:
            self.player_ids.remove(pid)
            del self.players[pid]
        # Adjust dealer index if needed
        if self.player_ids and self.dealer_index >= len(self.player_ids):
            self.dealer_index = 0
        return eliminated

    def rebuy_broke_players(self, starting_stack: int) -> list[str]:
        """Rebuy any player with 0 chips. Returns list of player IDs who rebought."""
        rebought = []
        for pid, player in self.players.items():
            if player.chips <= 0:
                player.chips = starting_stack
                player.rebuys += 1
                rebought.append(pid)
        return rebought

    def _get_dealer_id(self) -> str:
        return self.player_ids[self.dealer_index]

    def _get_small_blind_id(self) -> str:
        if self.is_heads_up:
            # Heads-up rule: dealer is small blind
            return self.player_ids[self.dealer_index]
        else:
            # Multi-way: player after dealer is small blind
            return self.player_ids[(self.dealer_index + 1) % self.num_players]

    def _get_big_blind_id(self) -> str:
        if self.is_heads_up:
            # Heads-up: non-dealer is big blind
            return self.player_ids[(self.dealer_index + 1) % 2]
        else:
            # Multi-way: player two seats after dealer is big blind
            return self.player_ids[(self.dealer_index + 2) % self.num_players]

    def _get_utg_index(self) -> int:
        """Get index of Under The Gun player (first to act preflop in multi-way)."""
        if self.is_heads_up:
            # Heads-up: dealer/SB acts first preflop
            return self.dealer_index
        else:
            # Multi-way: player after BB acts first
            return (self.dealer_index + 3) % self.num_players

    def start_new_hand(self) -> None:
        """Start a new hand."""
        self.hand_number += 1

        # Rotate dealer for this hand (except first hand)
        if self.hand_number > 1:
            self.dealer_index = (self.dealer_index + 1) % self.num_players

        self._hand_complete = False
        self._showdown_result = None

        # Reset deck and deal
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.last_raise_amount = self.big_blind
        self.betting_round = BettingRound.PREFLOP
        self.action_history = []

        # Reset players
        for player in self.players.values():
            player.reset_for_new_hand()

        # Deal hole cards
        for player in self.players.values():
            player.hole_cards = self.deck.deal(2)

        # Post blinds
        sb_player = self.players[self._get_small_blind_id()]
        bb_player = self.players[self._get_big_blind_id()]

        # Small blind
        sb_amount = min(self.small_blind, sb_player.chips)
        sb_player.chips -= sb_amount
        sb_player.current_bet = sb_amount
        sb_player.total_bet_this_hand = sb_amount
        self.pot += sb_amount
        if sb_player.chips == 0:
            sb_player.all_in = True

        # Big blind
        bb_amount = min(self.big_blind, bb_player.chips)
        bb_player.chips -= bb_amount
        bb_player.total_bet_this_hand = bb_amount
        bb_player.current_bet = bb_amount
        self.pot += bb_amount
        self.current_bet = bb_amount
        if bb_player.chips == 0:
            bb_player.all_in = True

        self.action_history.append(f"{sb_player.player_id} posts SB ${sb_amount}")
        self.action_history.append(f"{bb_player.player_id} posts BB ${bb_amount}")

        # Set first player to act preflop
        self.current_player_index = self._get_utg_index()

    def get_current_player(self) -> str:
        """Get the ID of the player whose turn it is."""
        return self.player_ids[self.current_player_index]

    def get_valid_actions(self, player_id: str) -> list[ActionType]:
        """Get list of valid actions for a player."""
        player = self.players[player_id]
        actions = []

        if player.folded or player.all_in:
            return []

        # Can always fold
        actions.append(ActionType.FOLD)

        # Check if can check (no bet to call)
        amount_to_call = self.current_bet - player.current_bet
        if amount_to_call == 0:
            actions.append(ActionType.CHECK)
        else:
            # Can call if we have chips
            if player.chips > 0:
                actions.append(ActionType.CALL)

        # Can raise if we have enough chips
        min_raise_to = self.current_bet + self.last_raise_amount
        if player.chips + player.current_bet > self.current_bet:
            # Can make at least a min raise or go all-in
            if player.chips + player.current_bet >= min_raise_to:
                actions.append(ActionType.RAISE)
            # Can always go all-in if we have chips
            if player.chips > 0:
                actions.append(ActionType.ALL_IN)

        return actions

    def get_min_raise(self) -> int:
        """Get minimum raise TO amount."""
        return self.current_bet + self.last_raise_amount

    def get_call_amount(self, player_id: str) -> int:
        """Get amount player needs to call."""
        player = self.players[player_id]
        return min(self.current_bet - player.current_bet, player.chips)

    def execute_action(self, player_id: str, action: PlayerAction) -> None:
        """Execute a player action."""
        if player_id != self.get_current_player():
            raise ValueError(f"Not {player_id}'s turn. Current player: {self.get_current_player()}")

        player = self.players[player_id]
        valid_actions = self.get_valid_actions(player_id)

        if action.action_type not in valid_actions:
            raise ValueError(
                f"Invalid action {action.action_type} for {player_id}. Valid: {valid_actions}"
            )

        if action.action_type == ActionType.FOLD:
            player.folded = True
            self.action_history.append(f"{player_id} folds")

        elif action.action_type == ActionType.CHECK:
            self.action_history.append(f"{player_id} checks")

        elif action.action_type == ActionType.CALL:
            call_amount = self.get_call_amount(player_id)
            player.chips -= call_amount
            player.current_bet += call_amount
            player.total_bet_this_hand += call_amount
            self.pot += call_amount
            if player.chips == 0:
                player.all_in = True
            self.action_history.append(f"{player_id} calls ${call_amount}")

        elif action.action_type == ActionType.RAISE:
            # Validate raise amount
            min_raise_to = self.get_min_raise()
            max_raise_to = player.chips + player.current_bet

            raise_to = action.amount
            if raise_to < min_raise_to:
                raise_to = min_raise_to
            if raise_to > max_raise_to:
                raise_to = max_raise_to

            # Calculate how much more player needs to put in
            amount_to_add = raise_to - player.current_bet
            player.chips -= amount_to_add
            player.total_bet_this_hand += amount_to_add
            self.pot += amount_to_add

            # Update raise tracking
            raise_amount = raise_to - self.current_bet
            self.last_raise_amount = raise_amount
            self.current_bet = raise_to
            player.current_bet = raise_to

            if player.chips == 0:
                player.all_in = True
                self.action_history.append(f"{player_id} raises to ${raise_to} (ALL-IN)")
            else:
                self.action_history.append(f"{player_id} raises to ${raise_to}")

        elif action.action_type == ActionType.ALL_IN:
            all_in_amount = player.chips
            total_bet = player.current_bet + all_in_amount

            if total_bet > self.current_bet:
                # This is a raise
                raise_amount = total_bet - self.current_bet
                if raise_amount >= self.last_raise_amount:
                    self.last_raise_amount = raise_amount
                self.current_bet = total_bet

            self.pot += all_in_amount
            player.total_bet_this_hand += all_in_amount
            player.current_bet = total_bet
            player.chips = 0
            player.all_in = True
            self.action_history.append(f"{player_id} ALL-IN ${all_in_amount}")

        player.has_acted_this_round = True

        # Check if hand is over or round is complete
        self._check_hand_state()

    def _check_hand_state(self) -> None:
        """Check if hand is over or betting round is complete."""
        active_players = [p for p in self.players.values() if not p.folded]

        # If only one player left, hand is over
        if len(active_players) == 1:
            self._end_hand_fold(active_players[0].player_id)
            return

        # Check if betting round is complete
        if self._is_betting_round_complete():
            self._advance_round()
        else:
            # Move to next player
            self._advance_to_next_player()

    def _is_betting_round_complete(self) -> bool:
        """Check if current betting round is complete."""
        active_players = [p for p in self.players.values() if not p.folded and not p.all_in]

        # If all remaining players are all-in, round is complete
        if len(active_players) == 0:
            return True

        # Check if all active players have acted and bets are equal
        for player in active_players:
            if not player.has_acted_this_round:
                return False
            if player.current_bet != self.current_bet:
                return False

        return True

    def _advance_to_next_player(self) -> None:
        """Move to the next player who can act."""
        for _ in range(self.num_players):
            self.current_player_index = (self.current_player_index + 1) % self.num_players
            player = self.players[self.player_ids[self.current_player_index]]
            if not player.folded and not player.all_in:
                return

    def _advance_round(self) -> None:
        """Advance to the next betting round or showdown."""
        # Reset for new round
        for player in self.players.values():
            player.reset_for_new_round()

        self.current_bet = 0
        self.last_raise_amount = self.big_blind

        if self.betting_round == BettingRound.PREFLOP:
            self.betting_round = BettingRound.FLOP
            self.community_cards.extend(self.deck.deal(3))
        elif self.betting_round == BettingRound.FLOP:
            self.betting_round = BettingRound.TURN
            self.community_cards.extend(self.deck.deal(1))
        elif self.betting_round == BettingRound.TURN:
            self.betting_round = BettingRound.RIVER
            self.community_cards.extend(self.deck.deal(1))
        elif self.betting_round == BettingRound.RIVER:
            self._end_hand_showdown()
            return

        # Check if we can continue betting (need at least one player not all-in)
        active_players = [p for p in self.players.values() if not p.folded and not p.all_in]
        if len(active_players) <= 1:
            # Run out remaining cards and go to showdown
            while len(self.community_cards) < 5:
                self.community_cards.extend(self.deck.deal(1))
            self._end_hand_showdown()
            return

        # Post-flop: first active player after dealer acts first
        # Find first active player starting from position after dealer
        first_to_act_index = (self.dealer_index + 1) % self.num_players
        for i in range(self.num_players):
            check_index = (first_to_act_index + i) % self.num_players
            player = self.players[self.player_ids[check_index]]
            if not player.folded and not player.all_in:
                self.current_player_index = check_index
                break

    def _end_hand_fold(self, winner_id: str) -> None:
        """End hand when opponent folds."""
        self._hand_complete = True
        winner = self.players[winner_id]
        winner.chips += self.pot

        # No showdown, no revealed hands
        self._showdown_result = ShowdownResult(
            winners=[winner_id],
            pot=self.pot,
            revealed_hands={},
            board=self.community_cards.copy(),
            hand_number=self.hand_number,
            winnings={winner_id: self.pot},
        )

    def _calculate_pots(self) -> list[Pot]:
        """Calculate main pot and any side pots based on player contributions."""
        # Get all non-folded players sorted by their total contribution
        contributions: list[tuple[str, int]] = []
        for pid, player in self.players.items():
            if not player.folded:
                contributions.append((pid, player.total_bet_this_hand))

        # Sort by contribution (ascending)
        contributions.sort(key=lambda x: x[1])

        pots: list[Pot] = []
        prev_level = 0

        for i, (_, level) in enumerate(contributions):
            if level > prev_level:
                # Calculate pot for this level
                # Each player at or above this level contributes (level - prev_level)
                # Plus folded players contribute up to their total bet
                pot_amount = 0

                # Contribution from non-folded players at this level and above
                for j in range(i, len(contributions)):
                    pot_amount += level - prev_level

                # Contribution from folded players (up to this level)
                for pid, player in self.players.items():
                    if player.folded:
                        # Folded player contributes min(their total bet, level) - prev_level
                        folded_contrib = min(player.total_bet_this_hand, level) - prev_level
                        if folded_contrib > 0:
                            pot_amount += folded_contrib

                # Eligible players are those at this contribution level and above
                eligible = [pid for pid, contrib in contributions[i:]]
                pots.append(Pot(amount=pot_amount, eligible_players=eligible))
                prev_level = level

        return pots

    def _end_hand_showdown(self) -> None:
        """End hand with showdown - handles side pots."""
        self._hand_complete = True

        active_players = [p for p in self.players.values() if not p.folded]

        # Evaluate hands
        hands_info: dict[str, tuple[list[Card], str, int]] = {}
        for player in active_players:
            rank, hand_name = evaluate_hand(player.hole_cards, self.community_cards)
            hands_info[player.player_id] = (player.hole_cards.copy(), hand_name, rank)

        # Calculate pots (main and side pots)
        pots = self._calculate_pots()

        # Distribute each pot to winner(s) among eligible players
        winnings: dict[str, int] = {pid: 0 for pid in self.player_ids}
        all_winners: set[str] = set()

        for pot in pots:
            # Find best hand among eligible players
            eligible_hands = {pid: info for pid, info in hands_info.items()
                             if pid in pot.eligible_players}

            if not eligible_hands:
                continue

            best_rank = min(info[2] for info in eligible_hands.values())
            pot_winners = [pid for pid, info in eligible_hands.items()
                          if info[2] == best_rank]

            # Split pot among winners
            share = pot.amount // len(pot_winners)
            remainder = pot.amount % len(pot_winners)

            for i, winner_id in enumerate(pot_winners):
                win_amount = share + (1 if i < remainder else 0)
                self.players[winner_id].chips += win_amount
                winnings[winner_id] += win_amount
                all_winners.add(winner_id)

        # Build revealed hands dict (without rank)
        revealed = {pid: (info[0], info[1]) for pid, info in hands_info.items()}

        # Filter to only players who actually won something
        winners = [pid for pid in all_winners if winnings[pid] > 0]

        self._showdown_result = ShowdownResult(
            winners=winners,
            pot=self.pot,
            revealed_hands=revealed,
            board=self.community_cards.copy(),
            hand_number=self.hand_number,
            winnings={pid: amt for pid, amt in winnings.items() if amt > 0},
        )

    def is_hand_complete(self) -> bool:
        """Check if the current hand is complete."""
        return self._hand_complete

    def get_showdown_result(self) -> ShowdownResult | None:
        """Get showdown result if hand is complete."""
        return self._showdown_result

    def _get_position_name(self, player_id: str) -> str:
        """Get position name for a player."""
        if player_id == self._get_dealer_id():
            return "BTN" if not self.is_heads_up else "BTN/SB"
        elif player_id == self._get_small_blind_id():
            return "SB"
        elif player_id == self._get_big_blind_id():
            return "BB"
        else:
            # Calculate relative position for multi-way
            player_index = self.player_ids.index(player_id)
            bb_index = self.player_ids.index(self._get_big_blind_id())
            seats_from_bb = (player_index - bb_index) % self.num_players

            if self.num_players <= 4:
                return "MP"  # Middle position
            elif seats_from_bb == 1:
                return "UTG"  # Under the gun
            elif player_index == (self.dealer_index - 1) % self.num_players:
                return "CO"  # Cutoff (seat before button)
            else:
                return "MP"

    def get_game_state_for_player(self, player_id: str) -> GameState:
        """Get game state from a player's perspective (hides opponent hole cards)."""
        player = self.players[player_id]
        player_index = self.player_ids.index(player_id)

        # Build opponent info list (clockwise from player)
        opponents: list[OpponentInfo] = []
        for i in range(1, self.num_players):
            opp_index = (player_index + i) % self.num_players
            opp_id = self.player_ids[opp_index]
            opp = self.players[opp_id]

            opponents.append(OpponentInfo(
                player_id=opp_id,
                chips=opp.chips,
                current_bet=opp.current_bet,
                folded=opp.folded,
                all_in=opp.all_in,
                position=self._get_position_name(opp_id),
                seat_number=i,
            ))

        num_active = sum(1 for p in self.players.values() if not p.folded)
        call_amount = self.get_call_amount(player_id)

        return GameState(
            hand_number=self.hand_number,
            betting_round=self.betting_round,
            pot=self.pot,
            community_cards=self.community_cards.copy(),
            hole_cards=player.hole_cards.copy(),
            player_id=player_id,
            player_chips=player.chips,
            player_current_bet=player.current_bet,
            opponents=opponents,
            num_active_players=num_active,
            total_players=self.num_players,
            current_bet=self.current_bet,
            min_raise=self.get_min_raise(),
            call_amount=call_amount,
            is_dealer=(player_id == self._get_dealer_id()),
            is_small_blind=(player_id == self._get_small_blind_id()),
            is_big_blind=(player_id == self._get_big_blind_id()),
            position_name=self._get_position_name(player_id),
            action_history=self.action_history.copy(),
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

    def get_context(self) -> dict[str, Any]:
        """Get context for memory updates."""
        return {
            "betting_round": self.betting_round.value,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "community_cards": [str(c) for c in self.community_cards],
            "hand_number": self.hand_number,
        }

    def get_chip_counts(self) -> dict[str, int]:
        """Get current chip counts for all players."""
        return {pid: p.chips for pid, p in self.players.items()}
