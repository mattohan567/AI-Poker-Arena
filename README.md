# AI Poker Arena

A multi-model LLM poker simulation platform that orchestrates Texas Hold'em games between different AI language models. Study how various AI models approach poker strategy, decision-making, and opponent adaptation.

## Features

- **Multi-LLM Support**: Run games between any combination of models from Anthropic, OpenAI, DeepSeek, Mistral, xAI (Grok), and Google
- **Full Texas Hold'em**: Complete poker engine with proper hand evaluation, all betting rounds, and side pot calculation
- **Opponent Profiling**: AI agents learn and track opponent tendencies (VPIP, PFR, aggression, fold rates) across hands
- **Cost Tracking**: Monitor API token usage and estimated costs per model
- **Game Logging**: SQLite database stores all game data for post-game analysis

## Supported Models

| Provider | Models |
|----------|--------|
| Anthropic | Claude Haiku, Sonnet, Opus |
| OpenAI | GPT-5, GPT-5-mini |
| DeepSeek | DeepSeek-chat |
| Mistral | Mistral-large, Mistral-small |
| xAI | Grok-4 (with reasoning options) |
| Google | Gemini-3-pro |

## Installation

1. Clone the repository:
```bash
git clone git@github.com:mattohan567/AI-Poker-Arena.git
cd AI-Poker-Arena
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your API keys by copying the example environment file:
```bash
cp .env.example .env
```

4. Edit `.env` and add your API keys for the providers you want to use.

## Usage

```bash
# 2-player game with 50 hands
python main.py -n 50 -p 2

# 4-player game with specific models
python main.py -n 30 -p 4 --models opus sonnet gpt5 deepseek

# 6-player game with 100 hands, quiet output
python main.py -n 100 -p 6 -q
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n, --hands` | Number of hands to play | 50 |
| `-p, --players` | Number of players (2-6) | 2 |
| `--models` | Space-separated list of models | Default varies by player count |
| `-q, --quiet` | Suppress detailed output | False |

## Project Structure

```
AI-Poker-Arena/
├── main.py              # Entry point, game runner
├── poker/               # Core poker engine
│   ├── engine.py        # Texas Hold'em game logic
│   ├── actions.py       # Action types and card models
│   └── hand_eval.py     # Hand evaluation using treys
├── agents/              # LLM-powered poker agents
│   ├── poker_agent.py   # Agent decision logic
│   └── memory.py        # Opponent profiling system
├── utils/               # Utilities
│   ├── database.py      # SQLite game logging
│   └── costs.py         # API cost tracking
├── requirements.txt     # Python dependencies
└── .env.example         # API key template
```

## License

MIT
