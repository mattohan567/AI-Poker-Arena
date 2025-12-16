"""LLM-powered poker agent."""

import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from agents.memory import OpponentMemory, ShowdownRecord
from poker.actions import ActionType, Card, PlayerAction
from poker.engine import GameState, ShowdownResult
from utils.costs import get_cost_tracker

logger = logging.getLogger(__name__)

# Models that don't support full JSON schema structured output
# These will use JSON mode with manual parsing instead
MODELS_WITHOUT_STRUCTURED_OUTPUT = [
    "deepseek-chat",
]

MAX_RETRIES = 2


class PokerDecision(BaseModel):
    """Structured output from LLM for poker decisions."""

    reasoning: str = Field(description="Step-by-step analysis of the situation")
    action: str = Field(description="Action to take: fold, check, call, raise, or all_in")
    amount: int = Field(default=0, description="Raise amount if applicable (total bet, not increment)")
    confidence: float = Field(ge=0, le=1, description="Confidence in decision (0-1)")
    opponent_read: str | None = Field(
        default=None, description="Optional observation about opponent tendencies"
    )


DECISION_PROMPT = """You are an expert poker player. Analyze this situation and decide your action.

## Your Hand
Hole Cards: {hole_cards}
Position: {position}

## Board
{board_display}

## Game State
Pot: ${pot} | Your Stack: ${your_stack}
Blinds: ${sb}/${bb}
Active Players: {num_active}
Current bet to call: ${to_call}
Minimum raise to: ${min_raise}

## Table (clockwise from you)
{table_display}

## Action History This Hand
{action_history}

## Opponent Profiles
{opponent_profiles}

## Valid Actions (YOU MUST CHOOSE ONE OF THESE)
{valid_actions}

## Instructions
1. Analyze the situation considering your hand strength, position, pot odds, and opponent tendencies
2. Consider what hands opponents could have based on their profiles and actions
3. You can bluff (bet/raise with weak hands to make opponents fold), especially against players with high fold rates - but use it selectively, getting caught is costly.
4. Choose your action - IMPORTANT: You MUST pick from the valid actions listed above. If CHECK is not listed, you cannot check. If CALL is not listed, you cannot call.
5. If raising, specify the total amount to raise TO (not the increment)

Respond with your reasoning and decision."""


SHOWDOWN_READ_PROMPT = """You just saw your opponent's hand at showdown. Generate a brief observation about their play style.

## Hand Summary
Your opponent showed: {opponent_cards}
Board: {board}
Result: {result}
Their actions this hand: {action_summary}

## Their Current Profile
{opponent_profile}

Based on this showdown, write a brief (1-2 sentence) observation about their tendencies or playing style that could help in future hands. Focus on actionable insights like "Willing to bluff with draws" or "Only shows down strong hands"."""


class PokerAgent:
    """LLM-powered poker agent with opponent memory."""

    def __init__(
        self,
        player_id: str,
        llm: BaseChatModel,
        memory: OpponentMemory,
        model_id: str = "unknown",
    ):
        self.player_id = player_id
        self.llm = llm
        self.memory = memory
        self.model_id = model_id.lower()
        self.last_reasoning = ""
        self.cost_tracker = get_cost_tracker()

        # Track if we were the preflop aggressor (for c-bet tracking)
        self._was_preflop_aggressor = False
        self._current_hand_number = 0

    def _needs_manual_json_parsing(self) -> bool:
        """Check if this model needs manual JSON parsing instead of structured output."""
        return any(m in self.model_id for m in MODELS_WITHOUT_STRUCTURED_OUTPUT)

    def _parse_json_response(self, content: str) -> PokerDecision:
        """Parse JSON from response content for models without structured output."""
        # Try to extract JSON from the response
        # First try to find JSON block in markdown
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Last resort: assume the whole content is JSON
                json_str = content

        data = json.loads(json_str)
        return PokerDecision(
            reasoning=data.get("reasoning", ""),
            action=data.get("action", "fold"),
            amount=data.get("amount", 0),
            confidence=data.get("confidence", 0.5),
            opponent_read=data.get("opponent_read"),
        )

    def get_action(
        self, game_state: GameState, valid_actions: list[ActionType]
    ) -> PlayerAction:
        """Get action from LLM with retry logic."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                prompt = self._build_prompt(game_state, valid_actions)

                if self._needs_manual_json_parsing():
                    # Use JSON mode with manual parsing for models without structured output
                    json_prompt = prompt + "\n\nRespond with a JSON object containing: reasoning (string), action (string: fold/check/call/raise/all_in), amount (number, for raises), confidence (number 0-1)."
                    response = self.llm.invoke(json_prompt)
                    parsed = self._parse_json_response(response.content)
                    self._log_usage(json_prompt, parsed)
                    self.last_reasoning = parsed.reasoning
                    return self._validate_or_fallback(parsed, valid_actions, game_state)
                else:
                    # Use structured output for models that support it
                    structured_llm = self.llm.with_structured_output(PokerDecision)
                    response = structured_llm.invoke(prompt)
                    self._log_usage(prompt, response)
                    self.last_reasoning = response.reasoning
                    return self._validate_or_fallback(response, valid_actions, game_state)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.player_id}: {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"All retries failed for {self.player_id}, using fallback")
                    return self._safe_fallback(valid_actions)

        return self._safe_fallback(valid_actions)

    def _build_prompt(self, game_state: GameState, valid_actions: list[ActionType]) -> str:
        """Build the decision prompt."""
        # Format hole cards
        hole_cards = " ".join(str(c) for c in game_state.hole_cards)

        # Format position
        position = game_state.position_name

        # Format board
        if game_state.community_cards:
            board_cards = " ".join(str(c) for c in game_state.community_cards)
            if len(game_state.community_cards) == 3:
                board_display = f"Flop: {board_cards}"
            elif len(game_state.community_cards) == 4:
                flop = " ".join(str(c) for c in game_state.community_cards[:3])
                turn = str(game_state.community_cards[3])
                board_display = f"Flop: {flop} | Turn: {turn}"
            else:
                flop = " ".join(str(c) for c in game_state.community_cards[:3])
                turn = str(game_state.community_cards[3])
                river = str(game_state.community_cards[4])
                board_display = f"Flop: {flop} | Turn: {turn} | River: {river}"
        else:
            board_display = "Preflop (no community cards)"

        # Format action history
        action_history = "\n".join(game_state.action_history) if game_state.action_history else "None yet"

        # Build table display (all opponents)
        table_lines = []
        for opp in game_state.opponents:
            status = ""
            if opp.folded:
                status = " (folded)"
            elif opp.all_in:
                status = " (all-in)"
            table_lines.append(
                f"  {opp.position}: {opp.player_id} - ${opp.chips}{status} [bet: ${opp.current_bet}]"
            )
        table_display = "\n".join(table_lines) if table_lines else "No opponents"

        # Build opponent profiles (only for active opponents)
        profile_lines = []
        for opp in game_state.opponents:
            if not opp.folded:
                profile = self.memory.get_profile(opp.player_id)
                profile_lines.append(f"### {opp.player_id} ({profile.hands_played} hands)")
                profile_lines.append(profile.get_summary())
                profile_lines.append("")
        opponent_profiles = "\n".join(profile_lines) if profile_lines else "No active opponents"

        # Format valid actions
        valid_actions_str = ", ".join(a.value.upper() for a in valid_actions)

        return DECISION_PROMPT.format(
            hole_cards=hole_cards,
            position=position,
            board_display=board_display,
            pot=game_state.pot,
            your_stack=game_state.player_chips,
            sb=game_state.small_blind,
            bb=game_state.big_blind,
            num_active=game_state.num_active_players,
            to_call=game_state.call_amount,
            min_raise=game_state.min_raise,
            table_display=table_display,
            action_history=action_history,
            opponent_profiles=opponent_profiles,
            valid_actions=valid_actions_str,
        )

    def _validate_or_fallback(
        self,
        response: PokerDecision,
        valid_actions: list[ActionType],
        game_state: GameState,
    ) -> PlayerAction:
        """Validate LLM response, fallback to safe action if invalid."""
        # Parse action string to ActionType
        action_str = response.action.lower().strip()
        action_map = {
            "fold": ActionType.FOLD,
            "check": ActionType.CHECK,
            "call": ActionType.CALL,
            "raise": ActionType.RAISE,
            "all_in": ActionType.ALL_IN,
            "all-in": ActionType.ALL_IN,
            "allin": ActionType.ALL_IN,
        }

        action_type = action_map.get(action_str)

        # If action not recognized or not valid, fallback
        if action_type is None or action_type not in valid_actions:
            logger.warning(
                f"{self.player_id} invalid action '{response.action}', using fallback"
            )
            if ActionType.CHECK in valid_actions:
                return PlayerAction(player_id=self.player_id, action_type=ActionType.CHECK)
            elif ActionType.CALL in valid_actions:
                return PlayerAction(player_id=self.player_id, action_type=ActionType.CALL)
            elif ActionType.FOLD in valid_actions:
                return PlayerAction(player_id=self.player_id, action_type=ActionType.FOLD)

        # Validate raise amount
        if action_type == ActionType.RAISE:
            # Clamp to valid range
            min_raise = game_state.min_raise
            max_raise = game_state.player_chips + game_state.player_current_bet

            amount = response.amount
            if amount < min_raise:
                amount = min_raise
            if amount > max_raise:
                # If they tried to raise more than they have, go all-in
                if ActionType.ALL_IN in valid_actions:
                    return PlayerAction(
                        player_id=self.player_id,
                        action_type=ActionType.ALL_IN,
                        amount=game_state.player_chips,
                    )
                amount = max_raise

            return PlayerAction(
                player_id=self.player_id, action_type=ActionType.RAISE, amount=amount
            )

        if action_type == ActionType.ALL_IN:
            return PlayerAction(
                player_id=self.player_id,
                action_type=ActionType.ALL_IN,
                amount=game_state.player_chips,
            )

        return PlayerAction(player_id=self.player_id, action_type=action_type)

    def _safe_fallback(self, valid_actions: list[ActionType]) -> PlayerAction:
        """Return the safest valid action."""
        if ActionType.CHECK in valid_actions:
            self.last_reasoning = "Fallback: checking"
            return PlayerAction(player_id=self.player_id, action_type=ActionType.CHECK)
        elif ActionType.FOLD in valid_actions:
            self.last_reasoning = "Fallback: folding"
            return PlayerAction(player_id=self.player_id, action_type=ActionType.FOLD)
        elif ActionType.CALL in valid_actions:
            self.last_reasoning = "Fallback: calling"
            return PlayerAction(player_id=self.player_id, action_type=ActionType.CALL)
        else:
            # Should never happen
            self.last_reasoning = "Fallback: no valid action found"
            return PlayerAction(player_id=self.player_id, action_type=ActionType.FOLD)

    def _log_usage(self, prompt: str, response: PokerDecision) -> None:
        """Log approximate token usage."""
        # Rough approximation: 4 chars per token
        input_tokens = len(prompt) // 4
        output_tokens = len(response.reasoning) // 4 + 50  # Add overhead for structure

        self.cost_tracker.log_call(self.model_id, input_tokens, output_tokens)

    def observe_action(
        self, player_id: str, action: PlayerAction, context: dict[str, Any]
    ) -> None:
        """Update memory with observed action."""
        if player_id == self.player_id:
            # Track if we raised preflop (for c-bet detection)
            if context.get("betting_round") == "preflop" and action.action_type in (
                ActionType.RAISE,
                ActionType.ALL_IN,
            ):
                self._was_preflop_aggressor = True
            return

        # Start tracking new hand if needed
        hand_number = context.get("hand_number", 0)
        if hand_number != self._current_hand_number:
            self._current_hand_number = hand_number
            self.memory.start_hand(player_id)
            self._was_preflop_aggressor = False

        # Determine if opponent is facing a raise or c-bet
        betting_round = context.get("betting_round", "preflop")

        # Check if this is a c-bet situation (we raised preflop, now it's flop)
        is_facing_cbet = (
            betting_round == "flop"
            and self._was_preflop_aggressor
            and context.get("current_bet", 0) > 0
        )

        # Check if facing a raise
        is_facing_raise = context.get("current_bet", 0) > 0

        # Determine if opponent was preflop aggressor
        # (Simplified: just check if they raised preflop)
        is_preflop_aggressor = context.get("opponent_raised_preflop", False)

        enriched_context = {
            **context,
            "is_facing_raise": is_facing_raise,
            "is_facing_cbet": is_facing_cbet,
            "is_preflop_aggressor": is_preflop_aggressor,
        }

        self.memory.update_from_action(player_id, action, enriched_context)

    def observe_showdown(self, result: ShowdownResult) -> None:
        """Update memory and generate read after showdown."""
        for player_id, (hole_cards, hand_name) in result.revealed_hands.items():
            if player_id == self.player_id:
                continue

            # Get action summary
            action_summary = self.memory.get_action_summary(player_id)

            # Create showdown record
            record = ShowdownRecord(
                hand_number=result.hand_number,
                hole_cards=hole_cards,
                board=result.board,
                pot_won=result.winnings.get(player_id, 0),
                action_summary=action_summary,
            )

            # Update memory with showdown
            self.memory.update_from_showdown(player_id, record)

            # End hand tracking
            self.memory.end_hand(player_id, went_to_showdown=True)

            # Generate LLM read
            note = self._generate_read(player_id, result, hole_cards, action_summary)
            if note:
                self.memory.add_note(player_id, note)

    def _generate_read(
        self,
        opponent_id: str,
        result: ShowdownResult,
        opponent_cards: list[Card],
        action_summary: str,
    ) -> str | None:
        """Generate LLM observation about opponent at showdown."""
        try:
            opponent_cards_str = " ".join(str(c) for c in opponent_cards)
            board_str = " ".join(str(c) for c in result.board)

            # Determine result text
            if opponent_id in result.winners:
                result_text = f"Won ${result.winnings.get(opponent_id, 0)}"
            else:
                result_text = "Lost"

            profile = self.memory.get_profile(opponent_id)

            prompt = SHOWDOWN_READ_PROMPT.format(
                opponent_cards=opponent_cards_str,
                board=board_str,
                result=result_text,
                action_summary=action_summary,
                opponent_profile=profile.get_summary(),
            )

            response = self.llm.invoke(prompt)

            # Log usage
            input_tokens = len(prompt) // 4
            output_tokens = len(response.content) // 4
            self.cost_tracker.log_call(self.model_id, input_tokens, output_tokens)

            # Extract the note (should be short)
            # Handle both string and list response formats (Gemini returns list)
            content = response.content
            if isinstance(content, list):
                content = "".join(str(c) for c in content)
            note = content.strip()
            if len(note) > 200:
                note = note[:200] + "..."

            return note

        except Exception as e:
            logger.warning(f"Failed to generate showdown read: {e}")
            return None

    def end_hand_no_showdown(self, opponent_id: str) -> None:
        """End hand tracking when no showdown occurred."""
        self.memory.end_hand(opponent_id, went_to_showdown=False)
