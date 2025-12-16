"""Hand evaluation using the treys library."""

from treys import Card as TreysCard
from treys import Evaluator

from poker.actions import Card

# Initialize evaluator once (singleton pattern)
_evaluator = Evaluator()

# Hand rank class names (treys uses 0-9)
HAND_CLASS_NAMES = {
    0: "Royal Flush",
    1: "Straight Flush",
    2: "Four of a Kind",
    3: "Full House",
    4: "Flush",
    5: "Straight",
    6: "Three of a Kind",
    7: "Two Pair",
    8: "Pair",
    9: "High Card",
}


def card_to_treys(card: Card) -> int:
    """Convert our Card to treys integer format."""
    treys_str = card.to_treys_str()
    return TreysCard.new(treys_str)


def cards_to_treys(cards: list[Card]) -> list[int]:
    """Convert list of Cards to treys integer format."""
    return [card_to_treys(c) for c in cards]


def evaluate_hand(hole_cards: list[Card], community_cards: list[Card]) -> tuple[int, str]:
    """
    Evaluate a poker hand.

    Args:
        hole_cards: Player's 2 hole cards
        community_cards: 3-5 community cards on the board

    Returns:
        Tuple of (rank, hand_name) where lower rank is better (1 = best possible)
    """
    if len(hole_cards) != 2:
        raise ValueError(f"Expected 2 hole cards, got {len(hole_cards)}")

    if len(community_cards) < 3:
        raise ValueError(f"Need at least 3 community cards, got {len(community_cards)}")

    treys_hole = cards_to_treys(hole_cards)
    treys_board = cards_to_treys(community_cards)

    # Treys evaluate returns a rank (lower is better, 1 is royal flush)
    rank = _evaluator.evaluate(treys_board, treys_hole)
    hand_class = _evaluator.get_rank_class(rank)
    hand_name = HAND_CLASS_NAMES.get(hand_class, "Unknown")

    return rank, hand_name


def get_hand_name(hole_cards: list[Card], community_cards: list[Card]) -> str:
    """Get human-readable hand name."""
    _, name = evaluate_hand(hole_cards, community_cards)
    return name


def compare_hands(
    hands: list[tuple[list[Card], list[Card]]], community_cards: list[Card]
) -> list[int]:
    """
    Compare multiple hands and return winner indices.

    Args:
        hands: List of (hole_cards, player_id) tuples - we only use hole_cards
        community_cards: Community cards on board

    Returns:
        List of indices of winning players (can be multiple in case of tie)
    """
    if not hands:
        return []

    ranks = []
    for hole_cards, _ in hands:
        rank, _ = evaluate_hand(hole_cards, community_cards)
        ranks.append(rank)

    # Lower rank is better
    best_rank = min(ranks)
    winners = [i for i, rank in enumerate(ranks) if rank == best_rank]

    return winners


def evaluate_hand_strength(hole_cards: list[Card], community_cards: list[Card]) -> float:
    """
    Get hand strength as a percentile (0-1, higher is better).

    This is useful for agents to understand relative hand strength.
    Treys ranks go from 1 (best) to 7462 (worst).
    """
    rank, _ = evaluate_hand(hole_cards, community_cards)
    # Convert to percentile (1.0 = best, 0.0 = worst)
    # Rank 1 -> 1.0, Rank 7462 -> 0.0
    return 1.0 - (rank - 1) / 7461
