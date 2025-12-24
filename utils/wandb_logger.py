"""Weights & Biases logging for LLM Poker Arena."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from utils.costs import CostTracker


@dataclass
class ModelStats:
    """Running statistics for a single model."""

    player_id: str
    model_id: str
    starting_stack: int
    hands_played: int = 0
    hands_won: int = 0
    total_invested: int = 0
    current_stack: int = 0
    rebuys: int = 0
    stack_history: list[int] = field(default_factory=list)

    @property
    def cumulative_win_rate(self) -> float:
        """Win rate over all hands played."""
        return self.hands_won / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def roi(self) -> float:
        """Return on investment: profit / total invested."""
        if self.total_invested == 0:
            return 0.0
        return self.profit / self.total_invested

    @property
    def profit(self) -> int:
        """Current profit (can be negative)."""
        return self.current_stack - self.total_invested


class WandbPokerLogger:
    """Weights & Biases logger for poker game metrics."""

    def __init__(self, enabled: bool = True):
        """Initialize the logger.

        Args:
            enabled: Whether W&B logging is enabled.
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None

        # Per-model stats
        self.model_stats: dict[str, ModelStats] = {}
        self.player_to_model: dict[str, str] = {}

        # Action pattern tracking: model -> {action -> count}
        self.action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Position-action: model -> {position -> {action -> count}}
        self.position_actions: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Street-action: model -> {street -> {action -> count}}
        self.street_actions: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Confidence calibration: model -> [(confidence, won)]
        self.confidence_outcomes: dict[str, list[tuple[float, bool]]] = defaultdict(list)

        # Hand tracking
        self.total_pots: list[int] = []
        self.showdown_count: int = 0
        self.total_hands: int = 0

        # Profile accuracy tracking
        self.profile_comparisons: list[dict] = []

        # Game config (for reference)
        self.game_id: int | None = None

    def init_run(
        self,
        project: str,
        entity: str | None,
        config: dict,
        name: str | None = None,
    ) -> None:
        """Initialize a W&B run with game configuration.

        Args:
            project: W&B project name.
            entity: W&B entity (team or username).
            config: Game configuration dict.
            name: Optional run name.
        """
        if not self.enabled:
            return

        self.run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=name,
            reinit="finish_previous",
        )

    def register_player(
        self,
        player_id: str,
        model_id: str,
        starting_stack: int,
    ) -> None:
        """Register a player/model for tracking.

        Args:
            player_id: Unique player identifier.
            model_id: LLM model identifier.
            starting_stack: Initial chip count.
        """
        if not self.enabled:
            return

        self.model_stats[player_id] = ModelStats(
            player_id=player_id,
            model_id=model_id,
            starting_stack=starting_stack,
            current_stack=starting_stack,
            total_invested=starting_stack,
        )
        self.player_to_model[player_id] = model_id

    def log_action(
        self,
        hand_num: int,
        player_id: str,
        model_id: str,
        action_type: str,
        amount: int,
        position: str,
        street: str,
        confidence: float | None,
        pot_size: int,
        stack_size: int,
    ) -> None:
        """Log a single action for pattern analysis.

        Args:
            hand_num: Current hand number.
            player_id: Player making the action.
            model_id: Model identifier.
            action_type: Type of action (fold, check, call, raise, all_in).
            amount: Bet/raise amount.
            position: Position at table (BTN, SB, BB, etc.).
            street: Betting round (preflop, flop, turn, river).
            confidence: LLM confidence in decision (0-1).
            pot_size: Current pot size.
            stack_size: Player's current stack.
        """
        if not self.enabled:
            return

        # Track action distributions
        self.action_counts[model_id][action_type] += 1
        self.position_actions[model_id][position][action_type] += 1
        self.street_actions[model_id][street][action_type] += 1

    def log_hand_result(
        self,
        hand_num: int,
        winners: list[str],
        pot: int,
        went_to_showdown: bool,
    ) -> None:
        """Log the result of a hand.

        Args:
            hand_num: Hand number.
            winners: List of winner player IDs.
            pot: Final pot size.
            went_to_showdown: Whether hand went to showdown.
        """
        if not self.enabled:
            return

        self.total_hands = hand_num
        self.total_pots.append(pot)

        if went_to_showdown:
            self.showdown_count += 1

        # Update hands won for winners
        for winner in winners:
            if winner in self.model_stats:
                self.model_stats[winner].hands_won += 1

        # Update hands played for all
        for stats in self.model_stats.values():
            stats.hands_played = hand_num

    def update_stacks(
        self,
        hand_num: int,
        chips: dict[str, int],
        rebuys: dict[str, int],
    ) -> None:
        """Update stack sizes after a hand.

        Args:
            hand_num: Current hand number.
            chips: Current chip counts by player.
            rebuys: Rebuy counts by player.
        """
        if not self.enabled:
            return

        for player_id, stack in chips.items():
            if player_id in self.model_stats:
                stats = self.model_stats[player_id]
                stats.current_stack = stack
                stats.stack_history.append(stack)

                # Update rebuys and total invested
                player_rebuys = rebuys.get(player_id, 0)
                if player_rebuys > stats.rebuys:
                    # New rebuy occurred
                    new_rebuys = player_rebuys - stats.rebuys
                    stats.total_invested += new_rebuys * stats.starting_stack
                    stats.rebuys = player_rebuys

    def log_hand_metrics(self, hand_num: int) -> None:
        """Log per-hand time series metrics to W&B.

        Args:
            hand_num: Current hand number (used as step).
        """
        if not self.enabled or self.run is None:
            return

        metrics = {"hand_number": hand_num}

        # Per-model metrics
        for player_id, stats in self.model_stats.items():
            prefix = player_id
            metrics[f"{prefix}/cumulative_win_rate"] = stats.cumulative_win_rate
            metrics[f"{prefix}/roi"] = stats.roi
            metrics[f"{prefix}/profit"] = stats.profit
            metrics[f"{prefix}/stack"] = stats.current_stack

            # Running average stack
            if stats.stack_history:
                metrics[f"{prefix}/avg_stack"] = sum(stats.stack_history) / len(stats.stack_history)

        # Aggregate metrics
        if self.total_pots:
            metrics["avg_pot_size"] = sum(self.total_pots) / len(self.total_pots)
            metrics["last_pot"] = self.total_pots[-1]

        if self.total_hands > 0:
            metrics["showdown_rate"] = self.showdown_count / self.total_hands

        self.run.log(metrics, step=hand_num)

    def log_confidence_outcome(
        self,
        model_id: str,
        confidence: float,
        won: bool,
    ) -> None:
        """Track confidence vs outcome for calibration.

        Args:
            model_id: Model identifier.
            confidence: Confidence level (0-1).
            won: Whether the player won the hand.
        """
        if not self.enabled:
            return

        self.confidence_outcomes[model_id].append((confidence, won))

    def log_profile_comparison(
        self,
        observer_id: str,
        observed_id: str,
        estimated_vpip: float,
        estimated_pfr: float,
        estimated_style: str,
        actual_vpip: float,
        actual_pfr: float,
        actual_style: str,
    ) -> None:
        """Log opponent profile prediction vs actual.

        Args:
            observer_id: Player who made the estimate.
            observed_id: Player being observed.
            estimated_vpip: Estimated VPIP.
            estimated_pfr: Estimated PFR.
            estimated_style: Estimated playing style.
            actual_vpip: Actual VPIP from database.
            actual_pfr: Actual PFR from database.
            actual_style: Actual playing style.
        """
        if not self.enabled:
            return

        self.profile_comparisons.append({
            "observer": observer_id,
            "observed": observed_id,
            "estimated_vpip": estimated_vpip,
            "estimated_pfr": estimated_pfr,
            "estimated_style": estimated_style,
            "actual_vpip": actual_vpip,
            "actual_pfr": actual_pfr,
            "actual_style": actual_style,
            "vpip_error": abs(estimated_vpip - actual_vpip),
            "pfr_error": abs(estimated_pfr - actual_pfr),
            "style_correct": estimated_style == actual_style,
        })

    def _compute_calibration_metrics(self, model_id: str) -> dict:
        """Compute confidence calibration metrics for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Dict with calibration metrics.
        """
        outcomes = self.confidence_outcomes.get(model_id, [])
        if not outcomes:
            return {}

        # Bin confidence levels
        bins = {
            "low": (0.0, 0.4),
            "medium": (0.4, 0.7),
            "high": (0.7, 1.01),
        }

        results = {}
        total = len(outcomes)

        for bin_name, (low, high) in bins.items():
            bin_outcomes = [(c, w) for c, w in outcomes if low <= c < high]
            if bin_outcomes:
                accuracy = sum(1 for _, w in bin_outcomes if w) / len(bin_outcomes)
                avg_conf = sum(c for c, _ in bin_outcomes) / len(bin_outcomes)
                results[f"{bin_name}_accuracy"] = accuracy
                results[f"{bin_name}_avg_confidence"] = avg_conf
                results[f"{bin_name}_count"] = len(bin_outcomes)

        # Expected Calibration Error (ECE)
        ece = 0.0
        for bin_name, (low, high) in bins.items():
            bin_outcomes = [(c, w) for c, w in outcomes if low <= c < high]
            if bin_outcomes:
                bin_size = len(bin_outcomes)
                accuracy = sum(1 for _, w in bin_outcomes if w) / bin_size
                avg_conf = sum(c for c, _ in bin_outcomes) / bin_size
                ece += (bin_size / total) * abs(accuracy - avg_conf)

        results["ece"] = ece
        return results

    def _create_action_tables(self) -> list:
        """Create W&B Tables for action distributions.

        Returns:
            List of W&B Table objects.
        """
        tables = []

        # Action distribution table
        action_data = []
        for model_id, actions in self.action_counts.items():
            total = sum(actions.values())
            for action, count in actions.items():
                pct = count / total if total > 0 else 0
                action_data.append([model_id, action, count, pct])

        if action_data:
            action_table = wandb.Table(
                columns=["model", "action", "count", "percentage"],
                data=action_data,
            )
            tables.append(("action_distribution", action_table))

        # Position-action table
        position_data = []
        for model_id, positions in self.position_actions.items():
            for position, actions in positions.items():
                for action, count in actions.items():
                    position_data.append([model_id, position, action, count])

        if position_data:
            position_table = wandb.Table(
                columns=["model", "position", "action", "count"],
                data=position_data,
            )
            tables.append(("position_actions", position_table))

        # Street-action table
        street_data = []
        for model_id, streets in self.street_actions.items():
            for street, actions in streets.items():
                for action, count in actions.items():
                    street_data.append([model_id, street, action, count])

        if street_data:
            street_table = wandb.Table(
                columns=["model", "street", "action", "count"],
                data=street_data,
            )
            tables.append(("street_actions", street_table))

        return tables

    def finalize(
        self,
        cost_tracker: "CostTracker",
        final_chips: dict[str, int],
        hands_won: dict[str, int],
        rebuys: dict[str, int],
        starting_stack: int,
        total_hands: int,
    ) -> None:
        """Finalize the run with summary metrics and close.

        Args:
            cost_tracker: Cost tracker with API usage data.
            final_chips: Final chip counts by player.
            hands_won: Hands won by player.
            rebuys: Rebuys by player.
            starting_stack: Starting stack size.
            total_hands: Total hands played.
        """
        if not self.enabled or self.run is None:
            return

        # Final performance by model
        for player_id, stats in self.model_stats.items():
            model_id = stats.model_id
            total_invested = starting_stack + (rebuys.get(player_id, 0) * starting_stack)
            profit = final_chips.get(player_id, 0) - total_invested
            roi = profit / total_invested if total_invested > 0 else 0
            win_rate = hands_won.get(player_id, 0) / total_hands if total_hands > 0 else 0

            self.run.summary[f"final/{player_id}/profit"] = profit
            self.run.summary[f"final/{player_id}/roi"] = roi
            self.run.summary[f"final/{player_id}/win_rate"] = win_rate
            self.run.summary[f"final/{player_id}/hands_won"] = hands_won.get(player_id, 0)
            self.run.summary[f"final/{player_id}/rebuys"] = rebuys.get(player_id, 0)
            self.run.summary[f"final/{player_id}/final_stack"] = final_chips.get(player_id, 0)
            self.run.summary[f"final/{player_id}/model_id"] = model_id

        # Action pattern summary
        for model_id, actions in self.action_counts.items():
            total = sum(actions.values())
            if total > 0:
                self.run.summary[f"actions/{model_id}/fold_rate"] = actions.get("fold", 0) / total
                self.run.summary[f"actions/{model_id}/call_rate"] = actions.get("call", 0) / total
                self.run.summary[f"actions/{model_id}/raise_rate"] = actions.get("raise", 0) / total

                # VPIP: any voluntary action preflop (call or raise, not check/fold)
                preflop = self.street_actions.get(model_id, {}).get("preflop", {})
                preflop_total = sum(preflop.values())
                if preflop_total > 0:
                    vpip = (preflop.get("call", 0) + preflop.get("raise", 0) + preflop.get("all_in", 0)) / preflop_total
                    pfr = (preflop.get("raise", 0) + preflop.get("all_in", 0)) / preflop_total
                    self.run.summary[f"actions/{model_id}/vpip"] = vpip
                    self.run.summary[f"actions/{model_id}/pfr"] = pfr

        # Cost metrics
        costs = cost_tracker.estimate_cost()
        total_cost = sum(costs.values())
        self.run.summary["costs/total"] = total_cost

        for model_id in cost_tracker.calls:
            model_cost = costs.get(model_id, 0)
            self.run.summary[f"costs/{model_id}/total"] = model_cost
            self.run.summary[f"costs/{model_id}/per_hand"] = model_cost / total_hands if total_hands > 0 else 0
            self.run.summary[f"costs/{model_id}/input_tokens"] = cost_tracker.input_tokens.get(model_id, 0)
            self.run.summary[f"costs/{model_id}/output_tokens"] = cost_tracker.output_tokens.get(model_id, 0)
            self.run.summary[f"costs/{model_id}/calls"] = cost_tracker.calls.get(model_id, 0)

            # Profit per dollar (find player using this model)
            for player_id, stats in self.model_stats.items():
                if stats.model_id == model_id and model_cost > 0:
                    profit = final_chips.get(player_id, 0) - stats.total_invested
                    self.run.summary[f"costs/{model_id}/profit_per_dollar"] = profit / model_cost

        # Confidence calibration
        for model_id in self.confidence_outcomes:
            calibration = self._compute_calibration_metrics(model_id)
            for key, value in calibration.items():
                self.run.summary[f"calibration/{model_id}/{key}"] = value

        # Profiling accuracy
        if self.profile_comparisons:
            vpip_errors = [c["vpip_error"] for c in self.profile_comparisons]
            pfr_errors = [c["pfr_error"] for c in self.profile_comparisons]
            style_correct = [c["style_correct"] for c in self.profile_comparisons]

            self.run.summary["profiling/vpip_mae"] = sum(vpip_errors) / len(vpip_errors)
            self.run.summary["profiling/pfr_mae"] = sum(pfr_errors) / len(pfr_errors)
            self.run.summary["profiling/style_accuracy"] = sum(style_correct) / len(style_correct)

        # Game summary
        self.run.summary["game/total_hands"] = total_hands
        self.run.summary["game/showdown_rate"] = self.showdown_count / total_hands if total_hands > 0 else 0
        if self.total_pots:
            self.run.summary["game/avg_pot_size"] = sum(self.total_pots) / len(self.total_pots)

        # Log tables
        for name, table in self._create_action_tables():
            self.run.log({name: table})

        # Finish run
        self.run.finish()


def compute_actual_stats_from_db(db, game_id: int, player_name: str) -> dict:
    """Compute actual poker stats from raw actions in database.

    Args:
        db: GameDatabase instance.
        game_id: Game ID to query.
        player_name: Player to compute stats for.

    Returns:
        Dict with actual VPIP, PFR, aggression, etc.
    """
    cursor = db.conn.cursor()

    # Get all preflop actions for this player
    cursor.execute("""
        SELECT action_type, hand_number
        FROM actions
        WHERE game_id = ? AND player_name = ? AND betting_round = 'preflop'
    """, (game_id, player_name))
    preflop_actions = cursor.fetchall()

    # Group by hand
    hands_preflop: dict[int, list[str]] = defaultdict(list)
    for row in preflop_actions:
        hands_preflop[row["hand_number"]].append(row["action_type"])

    total_hands = len(hands_preflop)
    if total_hands == 0:
        return {"vpip": 0.0, "pfr": 0.0, "aggression_factor": 0.0}

    # VPIP: hands where player voluntarily put chips in (call/raise/all_in)
    vpip_hands = sum(
        1 for actions in hands_preflop.values()
        if any(a in ("call", "raise", "all_in") for a in actions)
    )

    # PFR: hands where player raised preflop
    pfr_hands = sum(
        1 for actions in hands_preflop.values()
        if any(a in ("raise", "all_in") for a in actions)
    )

    # Get all actions for aggression factor
    cursor.execute("""
        SELECT action_type
        FROM actions
        WHERE game_id = ? AND player_name = ?
    """, (game_id, player_name))
    all_actions = [row["action_type"] for row in cursor.fetchall()]

    raises = sum(1 for a in all_actions if a in ("raise", "all_in"))
    calls = sum(1 for a in all_actions if a == "call")

    aggression = raises / calls if calls > 0 else float(raises) if raises > 0 else 0.0

    # Estimate style based on VPIP/PFR
    vpip = vpip_hands / total_hands
    pfr = pfr_hands / total_hands

    if vpip < 0.25 and aggression > 2:
        style = "TAG"  # Tight-Aggressive
    elif vpip >= 0.35 and aggression > 2:
        style = "LAG"  # Loose-Aggressive
    elif vpip < 0.20 and aggression < 1.5:
        style = "nit"
    elif vpip >= 0.40 and aggression < 1.5:
        style = "calling station"
    else:
        style = "unknown"

    return {
        "vpip": vpip,
        "pfr": pfr,
        "aggression_factor": aggression,
        "estimated_style": style,
    }
