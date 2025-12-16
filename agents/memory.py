"""Opponent memory and profiling system."""

from pydantic import BaseModel, Field

from poker.actions import ActionType, BettingRound, Card, PlayerAction


class ShowdownRecord(BaseModel):
    """Record of a hand that went to showdown."""

    hand_number: int
    hole_cards: list[Card]
    board: list[Card]
    pot_won: int
    action_summary: str  # e.g., "raised preflop, c-bet flop, barreled turn"


class OpponentProfile(BaseModel):
    """Profile tracking opponent tendencies."""

    player_id: str
    hands_played: int = 0

    # Aggression tracking
    vpip: float = 0.0  # Voluntarily put money in pot %
    pfr: float = 0.0  # Preflop raise %
    aggression_factor: float = 0.0  # (bets + raises) / calls
    cbet_frequency: float = 0.0  # Continuation bet %

    # Showdown data
    showdown_hands: list[ShowdownRecord] = Field(default_factory=list)
    wtsd: float = 0.0  # Went to showdown %

    # Betting patterns
    avg_bet_sizing: float = 0.0  # As % of pot
    fold_to_raise_rate: float = 0.0
    fold_to_cbet_rate: float = 0.0

    # LLM-generated reads (showdowns only)
    notes: list[str] = Field(default_factory=list)
    estimated_style: str = "unknown"  # TAG, LAG, nit, calling station

    def get_summary(self) -> str:
        """Get a text summary of the opponent profile."""
        if self.hands_played == 0:
            return "No data yet"

        recent_showdowns = ""
        if self.showdown_hands:
            recent = self.showdown_hands[-3:]  # Last 3
            showdown_strs = []
            for sd in recent:
                cards_str = " ".join(str(c) for c in sd.hole_cards)
                showdown_strs.append(f"{cards_str} ({sd.action_summary})")
            recent_showdowns = "; ".join(showdown_strs)

        notes_str = "; ".join(self.notes[-3:]) if self.notes else "None"

        return f"""Style: {self.estimated_style}
- VPIP: {self.vpip:.0%} | PFR: {self.pfr:.0%}
- C-bet frequency: {self.cbet_frequency:.0%}
- Fold to raise: {self.fold_to_raise_rate:.0%}
- Fold to c-bet: {self.fold_to_cbet_rate:.0%}
- Aggression factor: {self.aggression_factor:.1f}
- Avg bet sizing: {self.avg_bet_sizing:.0%} of pot
- WTSD: {self.wtsd:.0%}
- Recent showdowns: {recent_showdowns or "None"}
- Notes: {notes_str}"""


class _HandTracker:
    """Track stats for a single hand."""

    def __init__(self) -> None:
        self.vpip = False  # Did they voluntarily put money in?
        self.pfr = False  # Did they raise preflop?
        self.cbet_opportunity = False  # Were they the preflop aggressor?
        self.cbet = False  # Did they c-bet?
        self.faced_raise = False
        self.folded_to_raise = False
        self.faced_cbet = False
        self.folded_to_cbet = False
        self.bets_and_raises = 0
        self.calls = 0
        self.bet_sizes: list[float] = []  # As % of pot
        self.went_to_showdown = False
        self.actions: list[str] = []  # For action summary


class OpponentMemory:
    """Manages opponent profiles and stat tracking."""

    def __init__(self) -> None:
        self.profiles: dict[str, OpponentProfile] = {}
        self._hand_trackers: dict[str, _HandTracker] = {}

        # Running totals for stat calculation
        self._stats: dict[str, dict] = {}

    def get_profile(self, opponent_id: str) -> OpponentProfile:
        """Get or create profile for an opponent."""
        if opponent_id not in self.profiles:
            self.profiles[opponent_id] = OpponentProfile(player_id=opponent_id)
            self._stats[opponent_id] = {
                "hands": 0,
                "vpip_count": 0,
                "pfr_count": 0,
                "bets_raises": 0,
                "calls": 0,
                "cbet_opportunities": 0,
                "cbets": 0,
                "faced_raise_count": 0,
                "folded_to_raise_count": 0,
                "faced_cbet_count": 0,
                "folded_to_cbet_count": 0,
                "showdowns": 0,
                "bet_sizes": [],
            }
        return self.profiles[opponent_id]

    def start_hand(self, opponent_id: str) -> None:
        """Start tracking a new hand for an opponent."""
        self._hand_trackers[opponent_id] = _HandTracker()

    def update_from_action(
        self,
        opponent_id: str,
        action: PlayerAction,
        context: dict,
    ) -> None:
        """Update stats from an observed action."""
        profile = self.get_profile(opponent_id)

        # Ensure we have a hand tracker
        if opponent_id not in self._hand_trackers:
            self.start_hand(opponent_id)

        tracker = self._hand_trackers[opponent_id]
        betting_round = context.get("betting_round", "preflop")
        pot = context.get("pot", 0)
        is_facing_raise = context.get("is_facing_raise", False)
        is_facing_cbet = context.get("is_facing_cbet", False)
        is_preflop_aggressor = context.get("is_preflop_aggressor", False)

        # Track VPIP (any voluntary money preflop)
        if betting_round == "preflop":
            if action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
                tracker.vpip = True

            if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                tracker.pfr = True

        # Track c-bet
        if betting_round == "flop" and is_preflop_aggressor:
            tracker.cbet_opportunity = True
            if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                # First bet on flop by preflop aggressor is a c-bet
                tracker.cbet = True

        # Track fold to raise
        if is_facing_raise:
            tracker.faced_raise = True
            if action.action_type == ActionType.FOLD:
                tracker.folded_to_raise = True

        # Track fold to c-bet
        if is_facing_cbet:
            tracker.faced_cbet = True
            if action.action_type == ActionType.FOLD:
                tracker.folded_to_cbet = True

        # Track aggression
        if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            tracker.bets_and_raises += 1
            # Track bet sizing
            if pot > 0 and action.amount > 0:
                bet_pct = action.amount / pot
                tracker.bet_sizes.append(bet_pct)
        elif action.action_type == ActionType.CALL:
            tracker.calls += 1

        # Track actions for summary
        action_str = self._action_to_summary(action, betting_round)
        if action_str:
            tracker.actions.append(action_str)

    def _action_to_summary(self, action: PlayerAction, betting_round: str) -> str:
        """Convert action to summary string."""
        round_prefix = {
            "preflop": "preflop",
            "flop": "flop",
            "turn": "turn",
            "river": "river",
        }.get(betting_round, "")

        if action.action_type == ActionType.RAISE:
            return f"raised {round_prefix}"
        elif action.action_type == ActionType.ALL_IN:
            return f"all-in {round_prefix}"
        elif action.action_type == ActionType.CALL:
            return f"called {round_prefix}"
        elif action.action_type == ActionType.FOLD:
            return f"folded {round_prefix}"
        return ""

    def end_hand(self, opponent_id: str, went_to_showdown: bool = False) -> None:
        """End hand tracking and update running stats."""
        if opponent_id not in self._hand_trackers:
            return

        tracker = self._hand_trackers[opponent_id]
        stats = self._stats.get(opponent_id, {})

        # Update running totals
        stats["hands"] = stats.get("hands", 0) + 1

        if tracker.vpip:
            stats["vpip_count"] = stats.get("vpip_count", 0) + 1

        if tracker.pfr:
            stats["pfr_count"] = stats.get("pfr_count", 0) + 1

        stats["bets_raises"] = stats.get("bets_raises", 0) + tracker.bets_and_raises
        stats["calls"] = stats.get("calls", 0) + tracker.calls

        if tracker.cbet_opportunity:
            stats["cbet_opportunities"] = stats.get("cbet_opportunities", 0) + 1
            if tracker.cbet:
                stats["cbets"] = stats.get("cbets", 0) + 1

        if tracker.faced_raise:
            stats["faced_raise_count"] = stats.get("faced_raise_count", 0) + 1
            if tracker.folded_to_raise:
                stats["folded_to_raise_count"] = stats.get("folded_to_raise_count", 0) + 1

        if tracker.faced_cbet:
            stats["faced_cbet_count"] = stats.get("faced_cbet_count", 0) + 1
            if tracker.folded_to_cbet:
                stats["folded_to_cbet_count"] = stats.get("folded_to_cbet_count", 0) + 1

        if went_to_showdown:
            stats["showdowns"] = stats.get("showdowns", 0) + 1

        bet_sizes = stats.get("bet_sizes", [])
        bet_sizes.extend(tracker.bet_sizes)
        stats["bet_sizes"] = bet_sizes

        self._stats[opponent_id] = stats

        # Update profile with calculated stats
        self._recalculate_profile(opponent_id)

        # Clean up tracker
        del self._hand_trackers[opponent_id]

    def _recalculate_profile(self, opponent_id: str) -> None:
        """Recalculate profile stats from running totals."""
        profile = self.profiles[opponent_id]
        stats = self._stats[opponent_id]

        hands = stats.get("hands", 0)
        if hands == 0:
            return

        profile.hands_played = hands

        # VPIP
        profile.vpip = stats.get("vpip_count", 0) / hands

        # PFR
        profile.pfr = stats.get("pfr_count", 0) / hands

        # Aggression factor
        bets_raises = stats.get("bets_raises", 0)
        calls = stats.get("calls", 0)
        if calls > 0:
            profile.aggression_factor = bets_raises / calls
        else:
            profile.aggression_factor = bets_raises if bets_raises > 0 else 0.0

        # C-bet frequency
        cbet_opps = stats.get("cbet_opportunities", 0)
        if cbet_opps > 0:
            profile.cbet_frequency = stats.get("cbets", 0) / cbet_opps

        # Fold to raise
        faced_raise = stats.get("faced_raise_count", 0)
        if faced_raise > 0:
            profile.fold_to_raise_rate = stats.get("folded_to_raise_count", 0) / faced_raise

        # Fold to c-bet
        faced_cbet = stats.get("faced_cbet_count", 0)
        if faced_cbet > 0:
            profile.fold_to_cbet_rate = stats.get("folded_to_cbet_count", 0) / faced_cbet

        # WTSD
        profile.wtsd = stats.get("showdowns", 0) / hands

        # Average bet sizing
        bet_sizes = stats.get("bet_sizes", [])
        if bet_sizes:
            profile.avg_bet_sizing = sum(bet_sizes) / len(bet_sizes)

        # Estimate style
        profile.estimated_style = self._estimate_style(profile)

    def _estimate_style(self, profile: OpponentProfile) -> str:
        """Estimate player style from stats."""
        if profile.hands_played < 5:
            return "unknown (insufficient data)"

        # Tight = VPIP < 25%, Loose = VPIP > 35%
        # Aggressive = PFR/VPIP > 0.7, Passive = PFR/VPIP < 0.5

        tight = profile.vpip < 0.25
        loose = profile.vpip > 0.35

        pfr_vpip_ratio = profile.pfr / profile.vpip if profile.vpip > 0 else 0
        aggressive = pfr_vpip_ratio > 0.7 or profile.aggression_factor > 2
        passive = pfr_vpip_ratio < 0.5 and profile.aggression_factor < 1

        if tight and aggressive:
            return "TAG (tight-aggressive)"
        elif tight and passive:
            return "Nit (tight-passive)"
        elif loose and aggressive:
            return "LAG (loose-aggressive)"
        elif loose and passive:
            return "Calling Station (loose-passive)"
        elif profile.fold_to_raise_rate > 0.5:
            return "Weak-tight (folds to pressure)"
        else:
            return "Unknown style"

    def update_from_showdown(self, opponent_id: str, record: ShowdownRecord) -> None:
        """Update profile with showdown data."""
        profile = self.get_profile(opponent_id)
        profile.showdown_hands.append(record)

        # Keep only last 10 showdowns
        if len(profile.showdown_hands) > 10:
            profile.showdown_hands = profile.showdown_hands[-10:]

    def add_note(self, opponent_id: str, note: str) -> None:
        """Add an LLM-generated note about the opponent."""
        profile = self.get_profile(opponent_id)
        profile.notes.append(note)

        # Keep only last 10 notes
        if len(profile.notes) > 10:
            profile.notes = profile.notes[-10:]

    def get_action_summary(self, opponent_id: str) -> str:
        """Get action summary for current hand."""
        if opponent_id not in self._hand_trackers:
            return ""
        tracker = self._hand_trackers[opponent_id]
        return ", ".join(tracker.actions) if tracker.actions else "no significant actions"
