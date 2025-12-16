"""Data models for poker actions and cards."""

from enum import Enum
from pydantic import BaseModel, Field


class ActionType(Enum):
    """Valid poker actions."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


class BettingRound(Enum):
    """Poker betting rounds."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Card(BaseModel):
    """Represents a playing card."""

    rank: str = Field(description="Card rank: 2-9, T, J, Q, K, A")
    suit: str = Field(description="Card suit: h, d, c, s")

    def __str__(self) -> str:
        """Return human-readable card string."""
        suit_symbols = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
        return f"{self.rank}{suit_symbols.get(self.suit, self.suit)}"

    def __repr__(self) -> str:
        return f"Card(rank='{self.rank}', suit='{self.suit}')"

    def to_treys_str(self) -> str:
        """Convert to treys library format (e.g., 'As' for Ace of spades)."""
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_treys_str(cls, s: str) -> "Card":
        """Create Card from treys format string."""
        return cls(rank=s[0], suit=s[1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


class PlayerAction(BaseModel):
    """Represents a player's action."""

    player_id: str
    action_type: ActionType
    amount: int = Field(default=0, description="Bet/raise amount (0 for fold/check/call)")

    def __str__(self) -> str:
        if self.action_type == ActionType.RAISE:
            return f"{self.player_id}: {self.action_type.value} to ${self.amount}"
        elif self.action_type == ActionType.ALL_IN:
            return f"{self.player_id}: ALL-IN ${self.amount}"
        elif self.action_type == ActionType.CALL:
            return f"{self.player_id}: {self.action_type.value}"
        else:
            return f"{self.player_id}: {self.action_type.value}"
