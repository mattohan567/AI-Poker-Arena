# AI Poker Arena

A multi-model LLM poker simulation platform that orchestrates Texas Hold'em games between different AI language models. Study how various AI models approach poker strategy, decision-making, and opponent adaptation.

## Features

- **Multi-LLM Support**: Run games between any combination of models from Anthropic, OpenAI, DeepSeek, Mistral, xAI (Grok), and Google
- **Full Texas Hold'em**: Complete poker engine with proper hand evaluation, all betting rounds, and side pot calculation
- **Opponent Profiling**: AI agents learn and track opponent tendencies (VPIP, PFR, aggression, fold rates) across hands
- **Cost Tracking**: Monitor API token usage and estimated costs per model
- **Game Logging**: SQLite database stores all game data for post-game analysis
- **Weights & Biases Integration**: Optional experiment tracking with W&B for metrics visualization, confidence calibration, and cross-run comparisons
- **Analysis Notebooks**: Jupyter notebooks for deep-dive analysis including performance metrics, playing styles, and decision quality

## Tech Stack

- **Python 3.x** - Core language
- **LangChain / LangGraph** - LLM orchestration and agent framework
- **Pydantic** - Structured output validation
- **Treys** - Poker hand evaluation
- **SQLite** - Game data persistence
- **Weights & Biases** - Experiment tracking and visualization
- **Pandas** - Data analysis
- **Plotly** - Interactive charts and dashboards

## Supported Models

| Provider | Models |
|----------|--------|
| Anthropic | Claude Haiku, Sonnet, Opus |
| OpenAI | GPT-5, GPT-5-mini |
| DeepSeek | DeepSeek-chat |
| Mistral | Mistral-large, Mistral-small |
| xAI | Grok, Grok-noreason |
| Google | Gemini 2.5 Flash |

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

# Enable Weights & Biases logging
python main.py -n 50 -p 4 --wandb

# W&B with custom project and entity
python main.py -n 50 -p 4 --wandb --wandb-project my-poker-study --wandb-entity my-team
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n, --num-hands` | Number of hands to play | 50 |
| `-p, --players` | Number of players (2-6) | 2 |
| `--models` | Space-separated list of models | Default varies by player count |
| `-q, --quiet` | Suppress detailed output | False |
| `--wandb` | Enable Weights & Biases logging | False |
| `--wandb-project` | W&B project name | AIPoker |
| `--wandb-entity` | W&B entity (team or username) | None |

### Running Experiments

For large-scale experiments with round-robin matchups between all models:

```bash
# Run full experiment (all model pairs, 1000 hands each)
python run_experiments.py

# Resume an interrupted experiment
python run_experiments.py --resume

# Preview the experiment plan without running
python run_experiments.py --dry-run

# Check current progress
python run_experiments.py --status

# Run matchups in parallel (faster, requires more API capacity)
python run_experiments.py --parallel 3
```

### Generating Analysis Reports

After running experiments, generate an interactive HTML dashboard:

```bash
# Generate analysis report from game data
python analyze_results.py

# Custom output path
python analyze_results.py --output my_report.html
```

The report includes model rankings, head-to-head results, playing style analysis, cost efficiency metrics, and more.

## Project Structure

```
AI-Poker-Arena/
├── main.py              # Entry point, single game runner
├── run_experiments.py   # Round-robin experiment runner (parallel, resume)
├── analyze_results.py   # HTML dashboard generator
├── poker/               # Core poker engine
│   ├── engine.py        # Texas Hold'em game logic
│   ├── actions.py       # Action types and card models
│   └── hand_eval.py     # Hand evaluation using treys
├── agents/              # LLM-powered poker agents
│   ├── poker_agent.py   # Agent decision logic
│   └── memory.py        # Opponent profiling system
├── utils/               # Utilities
│   ├── database.py      # SQLite game logging
│   ├── costs.py         # API cost tracking
│   └── wandb_logger.py  # Weights & Biases integration
├── analysis/            # Post-game analysis notebooks
│   ├── poker_analysis.ipynb    # Performance & style analysis
│   └── hypothesis_tests.ipynb  # Statistical testing
├── docs/                # Static site / portfolio page
│   └── index.html       # Analysis report for GitHub Pages
├── data/                # Generated data (gitignored)
│   ├── games.db         # SQLite database
│   └── experiment_state.json   # Experiment progress
├── requirements.txt     # Python dependencies
└── .env.example         # API key template
```

## Analysis & Visualization

### Weights & Biases Tracking

Enable W&B logging to track experiments across runs:

- **Per-model metrics**: Win rates, ROI, profit, stack progression
- **Action analytics**: Fold/call/raise rates by position and street
- **Confidence calibration**: How well model confidence predicts outcomes
- **Opponent profiling accuracy**: Compare estimated vs actual player statistics
- **Cost analysis**: API costs per model and profit-per-dollar metrics

Set `WANDB_API_KEY` in your `.env` file or run `wandb login` before enabling.

### Jupyter Notebooks

The `analysis/` directory contains notebooks for post-game analysis:

- **poker_analysis.ipynb**: Performance deep-dives, playing style visualization, decision quality metrics, head-to-head matchups, and hand replays
- **hypothesis_tests.ipynb**: Statistical hypothesis testing for comparing model performance
