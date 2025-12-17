"""Main game runner for LLM Poker Arena."""

import argparse
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from agents.memory import OpponentMemory
from agents.poker_agent import PokerAgent
from poker.actions import ActionType
from poker.engine import PokerEngine
from poker.hand_eval import evaluate_hand, evaluate_hand_strength
from utils.costs import get_cost_tracker, reset_cost_tracker
from utils.database import get_database

# Load environment variables
load_dotenv()

# Available models configuration: (model_id, provider)
MODELS = {
    # Anthropic models
    "haiku": ("claude-haiku-4-5-20251001", "anthropic"),
    "sonnet": ("claude-sonnet-4-5-20250929", "anthropic"),
    "opus": ("claude-opus-4-5-20251101", "anthropic"),
    # OpenAI models (GPT-5 family)
    "gpt5": ("gpt-5.2", "openai"),
    "gpt5-mini": ("gpt-5-mini", "openai"),
    # DeepSeek models
    "deepseek": ("deepseek-chat", "deepseek"),
    # Mistral models
    "mistral": ("mistral-large-latest", "mistral"),
    "mistral-small": ("mistral-small-latest", "mistral"),
    # xAI (Grok) models
    "grok": ("grok-4-1-fast-reasoning", "xai"),  # Fast + reasoning, cheap
    "grok-noreason": ("grok-4-1-fast-non-reasoning", "xai"),  # Fast, no reasoning
    # Google Gemini models
    "gemini": ("gemini-3-pro-preview", "google"),
}

# Default player configurations for different game sizes (diverse providers)
DEFAULT_CONFIGS = {
    2: ["sonnet", "gpt5"],
    3: ["opus", "gpt5-mini", "grok"],
    4: ["opus", "sonnet", "gpt5", "deepseek"],
    5: ["opus", "sonnet", "gpt5-mini", "grok", "deepseek"],
    6: ["opus", "sonnet", "gpt5", "grok", "deepseek", "mistral"],
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class GameStats:
    """Track game statistics."""

    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.hand_results: list[dict] = []
        self.chip_history: list[dict] = []

    def log_decision(
        self, hand_num: int, player_id: str, action: str, reasoning: str
    ) -> None:
        """Log a decision."""
        self.decisions.append(
            {
                "hand": hand_num,
                "player": player_id,
                "action": action,
                "reasoning": reasoning[:200] if reasoning else "",
            }
        )

    def log_hand_result(self, hand_num: int, winners: list[str], pot: int) -> None:
        """Log hand result."""
        self.hand_results.append(
            {"hand": hand_num, "winners": winners, "pot": pot}
        )

    def log_chips(self, hand_num: int, chips: dict[str, int]) -> None:
        """Log chip counts."""
        self.chip_history.append({"hand": hand_num, **chips})

    def print_summary(self, rebuys: dict[str, int], starting_stack: int) -> None:
        """Print game summary."""
        print("\n" + "=" * 60)
        print("GAME SUMMARY")
        print("=" * 60)

        total_hands = len(self.hand_results)
        print(f"\nTotal hands played: {total_hands}")

        # Win counts
        win_counts: dict[str, int] = {}
        total_won: dict[str, int] = {}
        for result in self.hand_results:
            for winner in result["winners"]:
                win_counts[winner] = win_counts.get(winner, 0) + 1
                pot_share = result["pot"] // len(result["winners"])
                total_won[winner] = total_won.get(winner, 0) + pot_share

        print("\nHands won:")
        for player, wins in sorted(win_counts.items()):
            print(f"  {player}: {wins} ({wins/total_hands:.0%})")

        # Final chips
        if self.chip_history:
            final_chips = self.chip_history[-1]
            print("\nFinal chip counts:")
            for key, value in final_chips.items():
                if key != "hand":
                    print(f"  {key}: ${value}")

            # Calculate profit/loss (accounting for rebuys)
            print("\nProfit/Loss (accounting for rebuys):")
            for key, value in final_chips.items():
                if key != "hand":
                    player_rebuys = rebuys.get(key, 0)
                    total_invested = starting_stack + (player_rebuys * starting_stack)
                    profit = value - total_invested
                    rebuy_note = f" ({player_rebuys} rebuys)" if player_rebuys > 0 else ""
                    sign = "+" if profit >= 0 else ""
                    print(f"  {key}: {sign}${profit}{rebuy_note}")


def create_agent(player_name: str, model_key: str) -> PokerAgent:
    """Create a poker agent with the specified model."""
    model_id, provider = MODELS[model_key]

    if provider == "anthropic":
        llm = ChatAnthropic(model=model_id, temperature=0.7)
    elif provider == "openai":
        llm = ChatOpenAI(model=model_id, temperature=0.7)
    elif provider == "deepseek":
        llm = ChatOpenAI(
            model=model_id,
            temperature=0.7,
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
    elif provider == "mistral":
        llm = ChatOpenAI(
            model=model_id,
            temperature=0.7,
            base_url="https://api.mistral.ai/v1",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
    elif provider == "xai":
        llm = ChatOpenAI(
            model=model_id,
            temperature=0.7,
            base_url="https://api.x.ai/v1",
            api_key=os.getenv("XAI_API_KEY"),
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=90,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return PokerAgent(
        player_id=player_name,
        llm=llm,
        memory=OpponentMemory(),
        model_id=model_id,
    )


def run_game(
    num_hands: int = 50,
    num_players: int = 2,
    model_list: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """Run a poker game between multiple LLM players."""

    # Determine which models to use
    if model_list is None:
        model_list = DEFAULT_CONFIGS.get(num_players, DEFAULT_CONFIGS[2])

    # Ensure we have enough models for all players
    while len(model_list) < num_players:
        model_list.append(model_list[-1])  # Repeat last model
    model_list = model_list[:num_players]  # Trim to exact count

    # Check which providers we need
    providers_needed = set()
    for model_key in model_list:
        _, provider = MODELS[model_key]
        providers_needed.add(provider)

    # Check for required API keys
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "xai": "XAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    for provider in providers_needed:
        env_var = api_key_map.get(provider)
        if env_var and not os.getenv(env_var):
            logger.error(f"{env_var} not found in environment")
            sys.exit(1)

    # Reset cost tracker
    reset_cost_tracker()
    cost_tracker = get_cost_tracker()

    # Create agents and track player IDs
    agents: dict[str, PokerAgent] = {}
    player_ids: list[str] = []
    model_counts: dict[str, int] = {}

    for model_key in model_list:
        # Create unique player name
        model_counts[model_key] = model_counts.get(model_key, 0) + 1
        count = model_counts[model_key]
        if model_counts[model_key] == 1 and model_list.count(model_key) == 1:
            player_name = model_key
        else:
            player_name = f"{model_key}_{count}"

        agent = create_agent(player_name, model_key)
        agents[player_name] = agent
        player_ids.append(player_name)

    # Initialize engine
    starting_stack = 1000
    small_blind = 5
    big_blind = 10

    engine = PokerEngine(
        player_ids=player_ids,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
    )

    stats = GameStats()

    # Initialize database
    db = get_database()
    players_config = [{"name": pid, "model": agents[pid].model_id} for pid in player_ids]
    game_id = db.start_game(
        num_players=num_players,
        num_hands=num_hands,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        players_config=players_config,
    )

    # Register players in database
    for player_id in player_ids:
        db.add_player(game_id, player_id, agents[player_id].model_id, starting_stack)

    # Print header
    print("\n" + "=" * 60)
    print(f"LLM POKER ARENA - {num_players} Player Game (Game #{game_id})")
    print("=" * 60)
    print("Players:")
    for player_id in player_ids:
        model_id = agents[player_id].model_id
        print(f"  - {player_id}: {model_id}")
    print(f"Starting stack: ${starting_stack} each")
    print(f"Blinds: ${small_blind}/${big_blind}")
    print(f"Hands to play: {num_hands}")
    print("=" * 60 + "\n")

    # Main game loop (cash game style - no eliminations)
    for hand_num in range(1, num_hands + 1):

        engine.start_new_hand()
        dealer = engine._get_dealer_id()

        # Start hand in database
        hand_id = db.start_hand(game_id, hand_num, dealer)

        if verbose:
            print(f"\n--- Hand #{hand_num} ---")
            print(f"Dealer: {dealer}")

        # Track preflop raiser for c-bet detection
        preflop_raiser = None

        while not engine.is_hand_complete():
            current_player = engine.get_current_player()
            agent = agents[current_player]

            game_state = engine.get_game_state_for_player(current_player)
            valid_actions = engine.get_valid_actions(current_player)

            if verbose and game_state.betting_round.value != "preflop":
                board = " ".join(str(c) for c in game_state.community_cards)
                print(f"  Board: {board}")

            # Agent decides
            action = agent.get_action(game_state, valid_actions)

            if verbose:
                action_str = str(action)
                print(f"  {action_str}")
                if agent.last_reasoning:
                    # Print first 100 chars of reasoning
                    reason_preview = agent.last_reasoning[:100]
                    if len(agent.last_reasoning) > 100:
                        reason_preview += "..."
                    print(f"    Reasoning: {reason_preview}")

            # Log decision to stats and database
            stats.log_decision(
                hand_num, current_player, str(action.action_type.value), agent.last_reasoning
            )

            # Log action to database
            hole_cards = " ".join(str(c) for c in game_state.hole_cards)
            board_str = " ".join(str(c) for c in game_state.community_cards)

            # Compute enhanced metrics
            hand_strength = None
            hand_name = None
            if game_state.community_cards:
                hand_strength = evaluate_hand_strength(
                    game_state.hole_cards,
                    game_state.community_cards
                )
                _, hand_name = evaluate_hand(
                    game_state.hole_cards,
                    game_state.community_cards
                )

            # Pot odds (only when there's something to call)
            pot_odds = None
            if game_state.call_amount > 0:
                pot_odds = game_state.call_amount / (game_state.pot + game_state.call_amount)

            # SPR (stack-to-pot ratio)
            spr = None
            if game_state.pot > 0:
                spr = game_state.player_chips / game_state.pot

            # Effective stack
            opp_chips = [o.chips for o in game_state.opponents if not o.folded]
            effective_stack = min(game_state.player_chips, min(opp_chips)) if opp_chips else game_state.player_chips

            db.log_action(
                game_id=game_id,
                hand_id=hand_id,
                hand_number=hand_num,
                player_name=current_player,
                model_id=agent.model_id,
                betting_round=game_state.betting_round.value,
                action_type=action.action_type.value,
                amount=action.amount,
                reasoning=agent.last_reasoning,
                hole_cards=hole_cards,
                board=board_str,
                pot_before=game_state.pot,
                # New metrics
                confidence=agent.last_confidence,
                position=game_state.position_name,
                opponent_read=agent.last_opponent_read,
                latency_ms=agent.last_latency_ms,
                call_amount=game_state.call_amount,
                pot_size=game_state.pot,
                hand_strength=hand_strength,
                hand_name=hand_name,
                pot_odds=pot_odds,
                spr=spr,
                effective_stack=effective_stack,
            )

            # Track preflop raiser
            if (
                game_state.betting_round.value == "preflop"
                and action.action_type in (ActionType.RAISE, ActionType.ALL_IN)
            ):
                preflop_raiser = current_player

            # Execute action
            engine.execute_action(current_player, action)

            # Build context for memory update
            context = engine.get_context()
            context["opponent_raised_preflop"] = preflop_raiser == current_player

            # Notify all agents of the action
            for agent in agents.values():
                agent.observe_action(current_player, action, context)

        # Hand complete
        result = engine.get_showdown_result()

        if result:
            went_to_showdown = bool(result.revealed_hands)
            board_str = " ".join(str(c) for c in result.board)

            if result.revealed_hands:
                # Showdown occurred
                if verbose:
                    print(f"  --- Showdown ---")
                    for pid, (cards, hand_name) in result.revealed_hands.items():
                        cards_str = " ".join(str(c) for c in cards)
                        print(f"    {pid}: {cards_str} ({hand_name})")

                        # Log showdown to database
                        db.log_showdown(
                            game_id=game_id,
                            hand_id=hand_id,
                            hand_number=hand_num,
                            player_name=pid,
                            hole_cards=cards_str,
                            hand_name=hand_name,
                            amount_won=result.winnings.get(pid, 0),
                        )

                # Notify agents of showdown
                for agent in agents.values():
                    agent.observe_showdown(result)
            else:
                # No showdown (fold)
                # End hand tracking for opponent
                for player_id in engine.player_ids:
                    for agent in agents.values():
                        if player_id != agent.player_id:
                            agent.end_hand_no_showdown(player_id)

            # End hand in database
            db.end_hand(hand_id, result.pot, result.winners, board_str, went_to_showdown)

            if verbose:
                if len(result.winnings) == 1:
                    # Single winner - simple display
                    winner = list(result.winnings.keys())[0]
                    amount = list(result.winnings.values())[0]
                    print(f"  Winner: {winner} wins ${amount}")
                else:
                    # Multiple winners (side pots or split pot)
                    print(f"  Winners:")
                    for pid, amount in result.winnings.items():
                        print(f"    {pid} wins ${amount}")

            stats.log_hand_result(hand_num, result.winners, result.pot)

        # Auto-rebuy any broke players
        rebought = engine.rebuy_broke_players(starting_stack)
        for pid in rebought:
            if verbose:
                print(f"  💰 {pid} rebuys for ${starting_stack}")

        # Log chip counts
        chips = engine.get_chip_counts()
        stats.log_chips(hand_num, chips)

        # Periodic status update
        if hand_num % 10 == 0:
            print(f"\n=== After {hand_num} hands ===")
            for pid, chip_count in chips.items():
                player_rebuys = engine.players[pid].rebuys
                total_invested = starting_stack + (player_rebuys * starting_stack)
                profit = chip_count - total_invested
                rebuy_note = f" [{player_rebuys}R]" if player_rebuys > 0 else ""
                sign = "+" if profit >= 0 else ""
                print(f"  {pid}: ${chip_count} ({sign}${profit}){rebuy_note}")

            # Print opponent profile evolution (simplified for multi-player)
            if num_players <= 3:
                print("\n  Opponent Profiles:")
                for pid, agent in agents.items():
                    for opponent_id in engine.player_ids:
                        if opponent_id != pid:
                            profile = agent.memory.get_profile(opponent_id)
                            if profile.hands_played > 0:
                                print(f"\n  {pid}'s view of {opponent_id}:")
                                print(f"    Style: {profile.estimated_style}")
                                print(f"    VPIP: {profile.vpip:.0%} | PFR: {profile.pfr:.0%}")
                                print(f"    Aggression: {profile.aggression_factor:.1f}")

    # Game complete
    rebuys = {pid: player.rebuys for pid, player in engine.players.items()}
    stats.print_summary(rebuys, starting_stack)

    # Print cost summary
    print("\n" + cost_tracker.get_summary())

    # Save final results to database
    final_chips = engine.get_chip_counts()
    rebuys = {pid: player.rebuys for pid, player in engine.players.items()}
    hands_won = {}
    for result in stats.hand_results:
        for winner in result["winners"]:
            hands_won[winner] = hands_won.get(winner, 0) + 1

    # Build API usage dict
    api_usage = {}
    for model_id in cost_tracker.calls:
        api_usage[model_id] = {
            "calls": cost_tracker.calls[model_id],
            "input_tokens": cost_tracker.input_tokens[model_id],
            "output_tokens": cost_tracker.output_tokens[model_id],
            "cost": cost_tracker.estimate_cost().get(model_id, 0),
        }

    db.end_game(game_id, final_chips, hands_won, rebuys, starting_stack, api_usage)

    # Save opponent profiles - what each agent thinks about others
    for observer_id, agent in agents.items():
        for observed_id in player_ids:
            if observed_id != observer_id:
                profile = agent.memory.get_profile(observed_id)
                if profile.hands_played > 0:
                    observed_model = agents[observed_id].model_id
                    db.save_opponent_profile(
                        game_id=game_id,
                        observer_player=observer_id,
                        observer_model=agent.model_id,
                        observed_player=observed_id,
                        observed_model=observed_model,
                        profile={
                            "hands_played": profile.hands_played,
                            "vpip": profile.vpip,
                            "pfr": profile.pfr,
                            "aggression_factor": profile.aggression_factor,
                            "cbet_frequency": profile.cbet_frequency,
                            "fold_to_raise_rate": profile.fold_to_raise_rate,
                            "fold_to_cbet_rate": profile.fold_to_cbet_rate,
                            "wtsd": profile.wtsd,
                            "avg_bet_sizing": profile.avg_bet_sizing,
                            "estimated_style": profile.estimated_style,
                            "notes": profile.notes,
                        },
                    )

    print(f"\nGame data saved to database (Game #{game_id})")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Poker Arena - Multi-model poker simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -n 20 -p 3                    # 3-player game with default models
  python main.py -n 30 -p 4 --models opus sonnet haiku gpt4o
  python main.py -n 50 -p 6 --models opus sonnet sonnet haiku haiku gpt4o
  python main.py -n 20 -p 3 --models deepseek mistral grok

Available models:
  Anthropic: haiku, sonnet, opus
  OpenAI: gpt5, gpt5-mini
  DeepSeek: deepseek
  Mistral: mistral, mistral-small
  xAI (Grok): grok, grok-noreason
  Google: gemini
        """,
    )
    parser.add_argument(
        "-n", "--num-hands", type=int, default=50, help="Number of hands to play"
    )
    parser.add_argument(
        "-p", "--players", type=int, default=2, choices=[2, 3, 4, 5, 6],
        help="Number of players (2-6)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()),
        help="Specific models to use (will cycle if fewer than players)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce output verbosity"
    )
    args = parser.parse_args()

    run_game(
        num_hands=args.num_hands,
        num_players=args.players,
        model_list=args.models,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
