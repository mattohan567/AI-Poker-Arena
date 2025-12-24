"""Cost tracking for LLM API calls."""

from dataclasses import dataclass, field


# Pricing per 1M tokens (as of Dec 2025)
# Sources:
# - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
# - OpenAI: https://platform.openai.com/docs/pricing
# - DeepSeek: https://api-docs.deepseek.com/quick_start/pricing/
PRICING = {
    # OpenAI models (GPT-5 family)
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    # Legacy OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Claude 4.5 models (Nov 2025 pricing - Opus 4.5 is 67% cheaper than 4.1)
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    # DeepSeek V3.2-Exp (Sept 2025 pricing)
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    # Mistral models
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    # xAI (Grok) models
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    # Google Gemini models
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    # Legacy model names
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Legacy provider-based (fallback)
    "openai": {"input": 2.50, "output": 10.00},
    "anthropic": {"input": 3.00, "output": 15.00},
}


@dataclass
class CostTracker:
    """Track API usage and costs across models."""

    # Token counts by model
    calls: dict[str, int] = field(default_factory=dict)
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)

    def log_call(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        """Log an API call."""
        model_id = model_id.lower()
        if model_id not in self.calls:
            self.calls[model_id] = 0
            self.input_tokens[model_id] = 0
            self.output_tokens[model_id] = 0

        self.calls[model_id] += 1
        self.input_tokens[model_id] += input_tokens
        self.output_tokens[model_id] += output_tokens

    def estimate_cost(self) -> dict[str, float]:
        """Estimate total cost by model."""
        costs = {}
        for model_id in self.calls:
            pricing = PRICING.get(model_id, {"input": 0, "output": 0})
            input_cost = (self.input_tokens[model_id] / 1_000_000) * pricing["input"]
            output_cost = (self.output_tokens[model_id] / 1_000_000) * pricing["output"]
            costs[model_id] = input_cost + output_cost
        return costs

    def get_summary(self) -> str:
        """Get a formatted summary of usage and costs."""
        costs = self.estimate_cost()
        total_cost = sum(costs.values())

        lines = ["=== API Usage Summary ==="]
        for model_id in sorted(self.calls.keys()):
            lines.append(f"\n{model_id}:")
            lines.append(f"  Calls: {self.calls[model_id]}")
            lines.append(f"  Input tokens: {self.input_tokens[model_id]:,}")
            lines.append(f"  Output tokens: {self.output_tokens[model_id]:,}")
            lines.append(f"  Estimated cost: ${costs[model_id]:.4f}")

        lines.append(f"\nTotal estimated cost: ${total_cost:.4f}")
        return "\n".join(lines)

    def get_total_cost(self) -> float:
        """Get total estimated cost across all models."""
        return sum(self.estimate_cost().values())


# Global cost tracker instance
_global_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _global_tracker
    _global_tracker = CostTracker()
