#!/usr/bin/env python3
"""
Analysis Dashboard Generator for LLM Poker Arena

Generates a comprehensive static HTML report analyzing experiment results.

Usage:
    python analyze_results.py                    # Generate full report
    python analyze_results.py --output report.html  # Custom output path
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Database path
DB_PATH = Path("data/games.db")

# Model display names and colors
MODEL_COLORS = {
    "haiku": "#D4A574",
    "sonnet": "#CC785C",
    "opus": "#B8860B",
    "gpt5": "#10A37F",
    "gpt5-mini": "#74AA9C",
    "deepseek": "#4A90D9",
    "mistral": "#FF7000",
    "mistral-small": "#FFB366",
    "grok": "#1DA1F2",
    "grok-noreason": "#657786",
    "gemini": "#4285F4",
}


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def load_game_data() -> pd.DataFrame:
    """Load all game player results."""
    conn = get_connection()
    query = """
        SELECT
            gp.*,
            g.num_hands,
            g.starting_stack,
            g.small_blind,
            g.big_blind
        FROM game_players gp
        JOIN games g ON gp.game_id = g.id
        WHERE g.status = 'completed'
          AND g.num_hands = 1000
          AND g.num_players = 2
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate metrics
    df['total_invested'] = df['starting_chips'] + (df['rebuys'] * df['starting_chips'])
    df['profit'] = df['final_chips'] - df['total_invested']
    df['roi'] = df['profit'] / df['total_invested']
    df['bb_won'] = df['profit'] / df['big_blind']
    df['bb_per_100'] = (df['bb_won'] / df['num_hands']) * 100

    return df


def load_action_data() -> pd.DataFrame:
    """Load all action data."""
    conn = get_connection()
    query = """
        SELECT
            a.*,
            g.num_hands,
            g.big_blind
        FROM actions a
        JOIN games g ON a.game_id = g.id
        WHERE g.status = 'completed'
          AND g.num_hands = 1000
          AND g.num_players = 2
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_showdown_data() -> pd.DataFrame:
    """Load showdown data."""
    conn = get_connection()
    query = """
        SELECT s.*, g.big_blind
        FROM showdowns s
        JOIN games g ON s.game_id = g.id
        WHERE g.status = 'completed'
          AND g.num_hands = 1000
          AND g.num_players = 2
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_cost_data() -> pd.DataFrame:
    """Load API cost data."""
    conn = get_connection()
    query = "SELECT * FROM api_usage"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_opponent_profiles() -> pd.DataFrame:
    """Load opponent profile data filtered for experiment games."""
    conn = get_connection()
    query = """
        SELECT op.*
        FROM opponent_profiles op
        JOIN games g ON op.game_id = g.id
        WHERE g.status = 'completed'
          AND g.num_hands = 1000
          AND g.num_players = 2
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_hands_data() -> pd.DataFrame:
    """Load hands data for win tracking."""
    conn = get_connection()
    query = """
        SELECT h.*
        FROM hands h
        JOIN games g ON h.game_id = g.id
        WHERE g.status = 'completed'
          AND g.num_hands = 1000
          AND g.num_players = 2
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def extract_model_name(player_name: str) -> str:
    """Extract base model name from player name (e.g., 'sonnet_1' -> 'sonnet')."""
    parts = player_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return player_name


def compute_model_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall model rankings."""
    df = df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    rankings = df.groupby('model').agg({
        'profit': 'sum',
        'bb_per_100': 'mean',
        'roi': 'mean',
        'hands_won': 'sum',
        'num_hands': 'sum',
        'rebuys': 'sum',
        'game_id': 'count',
    }).reset_index()

    rankings.columns = ['model', 'total_profit', 'avg_bb_100', 'avg_roi',
                        'hands_won', 'total_hands', 'total_rebuys', 'games_played']
    rankings['win_rate'] = rankings['hands_won'] / rankings['total_hands']
    rankings = rankings.sort_values('avg_bb_100', ascending=False)

    return rankings


def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head results matrix based on hands won."""
    df = df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    # Group by game to get matchups with hands_won
    games = df.groupby('game_id').apply(
        lambda x: x[['model', 'hands_won']].to_dict('records'),
        include_groups=False
    ).reset_index(name='players')

    h2h_hands = defaultdict(lambda: defaultdict(lambda: {'won': 0, 'total': 0}))

    for _, row in games.iterrows():
        players = row['players']
        if len(players) == 2:
            m1, m2 = players[0]['model'], players[1]['model']
            h1, h2 = players[0]['hands_won'], players[1]['hands_won']
            total = h1 + h2

            # Accumulate hands won across all games against each opponent
            h2h_hands[m1][m2]['won'] += h1
            h2h_hands[m1][m2]['total'] += total
            h2h_hands[m2][m1]['won'] += h2
            h2h_hands[m2][m1]['total'] += total

    # Convert to win rates
    models = sorted(set(h2h_hands.keys()))
    matrix = []

    for m1 in models:
        row = {'model': m1}
        for m2 in models:
            if m1 == m2:
                row[m2] = None
            elif m2 in h2h_hands[m1] and h2h_hands[m1][m2]['total'] > 0:
                row[m2] = h2h_hands[m1][m2]['won'] / h2h_hands[m1][m2]['total']
            else:
                row[m2] = None
        matrix.append(row)

    return pd.DataFrame(matrix)


def compute_action_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute action statistics per model."""
    df = df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    # Action counts
    action_counts = df.groupby(['model', 'action_type']).size().unstack(fill_value=0)
    action_counts['total'] = action_counts.sum(axis=1)

    # Normalize to rates
    for col in action_counts.columns:
        if col != 'total':
            action_counts[f'{col}_rate'] = action_counts[col] / action_counts['total']

    return action_counts.reset_index()


def compute_position_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute position-based statistics."""
    df = df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    # Filter to preflop only for VPIP/PFR
    preflop = df[df['betting_round'] == 'preflop']

    stats = preflop.groupby(['model', 'position']).apply(
        lambda x: pd.Series({
            'vpip': (x['action_type'].isin(['call', 'raise', 'all_in'])).mean(),
            'pfr': (x['action_type'].isin(['raise', 'all_in'])).mean(),
            'count': len(x),
        })
    ).reset_index()

    return stats


def compute_confidence_calibration(df: pd.DataFrame, showdowns: pd.DataFrame) -> pd.DataFrame:
    """Compute confidence calibration data."""
    df = df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    # Join with showdown results to see if they won
    # This is a simplified version - in reality would need more sophisticated matching
    calibration_data = []

    for model in df['model'].unique():
        model_actions = df[df['model'] == model]

        if 'confidence' not in model_actions.columns:
            continue

        # Bin confidences
        for conf_bin in [(0, 0.33, 'low'), (0.33, 0.66, 'medium'), (0.66, 1.0, 'high')]:
            low, high, label = conf_bin
            bin_actions = model_actions[
                (model_actions['confidence'] >= low) &
                (model_actions['confidence'] < high)
            ]

            if len(bin_actions) > 0:
                # Approximate win rate from showdowns
                model_showdowns = showdowns[showdowns['player_name'].apply(extract_model_name) == model]
                win_rate = (model_showdowns['amount_won'] > 0).mean() if len(model_showdowns) > 0 else 0

                calibration_data.append({
                    'model': model,
                    'confidence_bin': label,
                    'avg_confidence': bin_actions['confidence'].mean(),
                    'count': len(bin_actions),
                    'win_rate': win_rate,
                })

    return pd.DataFrame(calibration_data)


def compute_cost_efficiency(game_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cost efficiency metrics."""
    # Aggregate by model
    game_df = game_df.copy()
    game_df['model'] = game_df['player_name'].apply(extract_model_name)

    profits = game_df.groupby('model')['profit'].sum().reset_index()

    # Get costs per model_id and map to model names
    model_id_to_name = {
        'claude-haiku-4-5-20251001': 'haiku',
        'claude-sonnet-4-5-20250929': 'sonnet',
        'claude-opus-4-5-20251101': 'opus',
        'gpt-5.2': 'gpt5',
        'gpt-5-mini': 'gpt5-mini',
        'deepseek-chat': 'deepseek',
        'mistral-large-latest': 'mistral',
        'mistral-small-latest': 'mistral-small',
        'grok-4-1-fast-reasoning': 'grok',
        'grok-4-1-fast-non-reasoning': 'grok-noreason',
        'gemini-3-pro-preview': 'gemini',
    }

    cost_df = cost_df.copy()
    cost_df['model'] = cost_df['model_id'].map(model_id_to_name)
    costs = cost_df.groupby('model').agg({
        'estimated_cost': 'sum',
        'total_calls': 'sum',
        'input_tokens': 'sum',
        'output_tokens': 'sum',
    }).reset_index()

    # Merge
    efficiency = profits.merge(costs, on='model', how='left')
    efficiency['profit_per_dollar'] = efficiency['profit'] / efficiency['estimated_cost'].replace(0, float('nan'))
    efficiency['cost_per_decision'] = efficiency['estimated_cost'] / efficiency['total_calls'].replace(0, float('nan'))

    return efficiency


def create_rankings_chart(rankings: pd.DataFrame) -> str:
    """Create model rankings bar chart."""
    fig = go.Figure()

    colors = [MODEL_COLORS.get(m, '#888888') for m in rankings['model']]

    fig.add_trace(go.Bar(
        x=rankings['model'],
        y=rankings['avg_bb_100'],
        marker_color=colors,
        text=[f"{v:.2f}" for v in rankings['avg_bb_100']],
        textposition='outside',
    ))

    fig.update_layout(
        title="Model Rankings by BB/100",
        xaxis_title="Model",
        yaxis_title="BB/100 (Big Blinds per 100 hands)",
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
        yaxis=dict(range=[min(rankings['avg_bb_100'].min() * 1.3, rankings['avg_bb_100'].min() - 5),
                          max(rankings['avg_bb_100'].max() * 1.3, rankings['avg_bb_100'].max() + 5)]),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_h2h_heatmap(h2h: pd.DataFrame) -> str:
    """Create head-to-head heatmap."""
    models = h2h['model'].tolist()
    z_data = h2h[models].values

    # Custom text for hover
    text = []
    for i, m1 in enumerate(models):
        row = []
        for j, m2 in enumerate(models):
            if i == j:
                row.append("")
            elif z_data[i][j] is not None:
                row.append(f"{m1} vs {m2}: {z_data[i][j]*100:.1f}% win rate")
            else:
                row.append("No data")
        text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=models,
        y=models,
        text=text,
        texttemplate="%{z:.0%}",
        colorscale=[[0, '#ff6b6b'], [0.5, '#ffffff'], [1, '#51cf66']],
        zmid=0.5,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        title="Head-to-Head Win Rates (Row vs Column)",
        xaxis_title="Opponent",
        yaxis_title="Model",
        template="plotly_white",
        height=500,
        width=700,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_roi_chart(rankings: pd.DataFrame) -> str:
    """Create ROI comparison chart."""
    fig = go.Figure()

    colors = [MODEL_COLORS.get(m, '#888888') for m in rankings['model']]

    fig.add_trace(go.Bar(
        x=rankings['model'],
        y=rankings['avg_roi'] * 100,
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in rankings['avg_roi']],
        textposition='outside',
    ))

    roi_values = rankings['avg_roi'] * 100
    fig.update_layout(
        title="Return on Investment by Model",
        xaxis_title="Model",
        yaxis_title="ROI (%)",
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
        yaxis=dict(range=[min(roi_values.min() * 1.3, roi_values.min() - 10),
                          max(roi_values.max() * 1.3, roi_values.max() + 10)]),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_action_distribution_chart(action_stats: pd.DataFrame) -> str:
    """Create action distribution chart."""
    fig = go.Figure()

    action_types = ['fold', 'check', 'call', 'raise', 'all_in']

    for action in action_types:
        rate_col = f'{action}_rate'
        if rate_col in action_stats.columns:
            fig.add_trace(go.Bar(
                name=action.capitalize(),
                x=action_stats['model'],
                y=action_stats[rate_col] * 100,
            ))

    fig.update_layout(
        title="Action Distribution by Model",
        xaxis_title="Model",
        yaxis_title="Action Rate (%)",
        barmode='stack',
        template="plotly_white",
        height=400,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_cost_efficiency_chart(efficiency: pd.DataFrame) -> str:
    """Create cost efficiency scatter plot."""
    # Drop rows with NaN values to avoid plotly errors
    eff_clean = efficiency.dropna(subset=['total_calls', 'estimated_cost', 'profit'])
    fig = px.scatter(
        eff_clean,
        x='estimated_cost',
        y='profit',
        size='total_calls',
        color='model',
        color_discrete_map=MODEL_COLORS,
        hover_data=['profit_per_dollar'],
        title="Cost vs Profit by Model",
    )

    fig.update_layout(
        xaxis_title="Total API Cost ($)",
        yaxis_title="Total Profit (chips)",
        template="plotly_white",
        height=450,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_profit_per_dollar_chart(efficiency: pd.DataFrame) -> str:
    """Create profit per dollar chart."""
    eff_sorted = efficiency.sort_values('profit_per_dollar', ascending=False)
    colors = [MODEL_COLORS.get(m, '#888888') for m in eff_sorted['model']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=eff_sorted['model'],
        y=eff_sorted['profit_per_dollar'],
        marker_color=colors,
        text=[f"{v:.0f}" if pd.notna(v) else "N/A" for v in eff_sorted['profit_per_dollar']],
        textposition='outside',
    ))

    # Calculate y-axis range to accommodate text labels
    valid_vals = eff_sorted['profit_per_dollar'].dropna()
    y_min = valid_vals.min() if len(valid_vals) > 0 else 0
    y_max = valid_vals.max() if len(valid_vals) > 0 else 100
    y_padding = max(abs(y_max), abs(y_min)) * 0.2

    fig.update_layout(
        title="Profit per API Dollar (Cost Efficiency)",
        xaxis_title="Model",
        yaxis_title="Chips Profit per $1 API Cost",
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
        yaxis=dict(range=[y_min - y_padding, y_max + y_padding]),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_playing_style_radar(action_stats: pd.DataFrame, rankings: pd.DataFrame) -> str:
    """Create playing style radar charts."""
    models_to_show = action_stats['model'].head(6).tolist()

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'polar'}]*3]*2,
        subplot_titles=models_to_show,
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    categories = ['Fold', 'Check', 'Call', 'Raise', 'Win']

    for idx, (_, row) in enumerate(action_stats.head(6).iterrows()):
        model = row['model']
        model_rank = rankings[rankings['model'] == model]
        win_rate = model_rank['win_rate'].values[0] if len(model_rank) > 0 else 0

        values = [
            row.get('fold_rate', 0) * 100,
            row.get('check_rate', 0) * 100,
            row.get('call_rate', 0) * 100,
            row.get('raise_rate', 0) * 100,
            win_rate * 100,
        ]

        r = idx // 3 + 1
        c = idx % 3 + 1

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=model,
            line_color=MODEL_COLORS.get(model, '#888888'),
        ), row=r, col=c)

    fig.update_layout(
        title="Playing Style Profiles (Top 6 Models)",
        template="plotly_white",
        height=700,
        showlegend=False,
        margin=dict(t=80, b=40, l=40, r=40),
    )

    # Update annotations (subplot titles) to be positioned higher
    for annotation in fig['layout']['annotations']:
        annotation['y'] = annotation['y'] + 0.02
        annotation['font'] = dict(size=12)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_profit_chart(rankings: pd.DataFrame) -> str:
    """Create total profit horizontal bar chart."""
    sorted_rankings = rankings.sort_values('total_profit', ascending=True)
    colors = ['#51cf66' if p > 0 else '#ff6b6b' for p in sorted_rankings['total_profit']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_rankings['total_profit'],
        y=sorted_rankings['model'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:,.0f}" for v in sorted_rankings['total_profit']],
        textposition='outside',
    ))

    # Calculate x-axis range to accommodate text labels
    x_min = sorted_rankings['total_profit'].min()
    x_max = sorted_rankings['total_profit'].max()
    x_padding = max(abs(x_max), abs(x_min)) * 0.25

    fig.update_layout(
        title="Total Profit by Model",
        xaxis_title="Total Profit (chips)",
        yaxis_title="Model",
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=50, l=80, r=80),
        xaxis=dict(range=[x_min - x_padding, x_max + x_padding]),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def compute_vpip_pfr(action_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VPIP and PFR stats per model."""
    df = action_df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    preflop = df[df['betting_round'] == 'preflop']

    stats = preflop.groupby('model').apply(
        lambda x: pd.Series({
            'vpip': (x['action_type'].isin(['call', 'raise', 'all_in']).sum() / len(x) * 100) if len(x) > 0 else 0,
            'pfr': (x['action_type'].isin(['raise', 'all_in']).sum() / len(x) * 100) if len(x) > 0 else 0,
            'hands': len(x),
        }),
        include_groups=False
    ).reset_index()

    return stats


def create_vpip_pfr_chart(vpip_pfr: pd.DataFrame) -> str:
    """Create VPIP vs PFR scatter plot with quadrant labels."""
    fig = px.scatter(
        vpip_pfr,
        x='vpip',
        y='pfr',
        text='model',
        size='hands',
        title='Playing Style: VPIP vs PFR',
        labels={'vpip': 'VPIP %', 'pfr': 'PFR %'},
        color_discrete_sequence=['#667eea'],
    )

    # Add quadrant lines
    fig.add_hline(y=25, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=35, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    fig.add_annotation(x=15, y=45, text="TAG", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=55, y=45, text="LAG", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=15, y=10, text="Nit", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=55, y=10, text="Calling Station", showarrow=False, font=dict(size=12, color="gray"))

    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_white", height=450)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def compute_opponent_profile_summary(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate opponent profiles by observed model."""
    if len(profiles_df) == 0:
        return pd.DataFrame()

    # Map model IDs to short names
    model_id_to_name = {
        'claude-haiku-4-5-20251001': 'haiku',
        'claude-sonnet-4-5-20250929': 'sonnet',
        'claude-opus-4-5-20251101': 'opus',
        'gpt-5.2': 'gpt5',
        'gpt-5-mini': 'gpt5-mini',
        'deepseek-chat': 'deepseek',
        'mistral-large-latest': 'mistral',
        'mistral-small-latest': 'mistral-small',
        'grok-4-1-fast-reasoning': 'grok',
        'grok-4-1-fast-non-reasoning': 'grok-noreason',
        'gemini-3-pro-preview': 'gemini',
    }

    df = profiles_df.copy()
    df['observer'] = df['observer_model'].map(model_id_to_name).fillna(df['observer_model'])
    df['observed'] = df['observed_model'].map(model_id_to_name).fillna(df['observed_model'])

    summary = df.groupby('observed').agg({
        'vpip': 'mean',
        'pfr': 'mean',
        'aggression_factor': 'mean',
        'fold_to_raise_rate': 'mean',
        'wtsd': 'mean',
        'estimated_style': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
    }).round(2).reset_index()

    summary.columns = ['Model', 'Avg VPIP', 'Avg PFR', 'Aggression', 'Fold to Raise', 'WTSD', 'Perceived Style']

    return summary


def create_opponent_profile_table_html(profile_summary: pd.DataFrame) -> str:
    """Create HTML table for opponent profiles."""
    if len(profile_summary) == 0:
        return "<p>No opponent profile data available.</p>"

    rows = ""
    for _, row in profile_summary.sort_values('Avg VPIP', ascending=False).iterrows():
        rows += f"""
        <tr>
            <td><strong>{row['Model']}</strong></td>
            <td>{row['Avg VPIP']:.1%}</td>
            <td>{row['Avg PFR']:.1%}</td>
            <td>{row['Aggression']:.2f}</td>
            <td>{row['Fold to Raise']:.1%}</td>
            <td>{row['WTSD']:.1%}</td>
            <td>{row['Perceived Style']}</td>
        </tr>
        """

    return f"""
    <table>
        <tr>
            <th>Model</th>
            <th>VPIP</th>
            <th>PFR</th>
            <th>Aggression</th>
            <th>Fold to Raise</th>
            <th>WTSD</th>
            <th>Perceived Style</th>
        </tr>
        {rows}
    </table>
    """


def compute_confidence_stats(action_df: pd.DataFrame, hands_df: pd.DataFrame) -> pd.DataFrame:
    """Compute confidence calibration stats."""
    if 'confidence' not in action_df.columns or action_df['confidence'].isna().all():
        return pd.DataFrame()

    df = action_df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)

    # Parse winners from hands
    def parse_winners(winners_str):
        try:
            return json.loads(winners_str) if winners_str else []
        except:
            return []

    hands_df = hands_df.copy()
    hands_df['winners_list'] = hands_df['winners'].apply(parse_winners)

    # Link actions to hand outcomes
    conf_wins = []
    for _, hand in hands_df.iterrows():
        winners = hand['winners_list']
        hand_actions = df[(df['game_id'] == hand['game_id']) &
                          (df['hand_number'] == hand['hand_number']) &
                          (df['confidence'].notna())]
        for _, action in hand_actions.iterrows():
            conf_wins.append({
                'model': action['model'],
                'confidence': action['confidence'],
                'won': 1 if action['player_name'] in winners else 0,
            })

    if not conf_wins:
        return pd.DataFrame()

    conf_df = pd.DataFrame(conf_wins)

    # Aggregate by model
    stats = conf_df.groupby('model').agg({
        'confidence': 'mean',
        'won': 'mean',
    }).reset_index()
    stats.columns = ['model', 'avg_confidence', 'win_rate']
    stats['win_rate_pct'] = stats['win_rate'] * 100

    return stats


def create_confidence_chart(conf_stats: pd.DataFrame) -> str:
    """Create confidence calibration scatter plot."""
    if len(conf_stats) == 0:
        return "<p>No confidence data available.</p>"

    fig = px.scatter(
        conf_stats,
        x='avg_confidence',
        y='win_rate_pct',
        text='model',
        title='Confidence Calibration: Avg Confidence vs Actual Win Rate',
        labels={'avg_confidence': 'Average Confidence', 'win_rate_pct': 'Actual Win Rate (%)'},
        color_discrete_sequence=['#667eea'],
    )

    # Add diagonal line for perfect calibration
    fig.add_shape(
        type='line',
        x0=0, y0=0, x1=1, y1=100,
        line=dict(color='gray', dash='dash'),
    )
    fig.add_annotation(x=0.7, y=85, text="Perfect Calibration", showarrow=False,
                       font=dict(size=10, color='gray'))

    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_white", height=400)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def compute_latency_stats(action_df: pd.DataFrame) -> pd.DataFrame:
    """Compute latency statistics per model."""
    if 'latency_ms' not in action_df.columns or action_df['latency_ms'].isna().all():
        return pd.DataFrame()

    df = action_df.copy()
    df['model'] = df['player_name'].apply(extract_model_name)
    df = df[df['latency_ms'].notna() & (df['latency_ms'] > 0)]

    if len(df) == 0:
        return pd.DataFrame()

    stats = df.groupby('model')['latency_ms'].agg(['mean', 'median', 'std', 'count']).round(0).reset_index()
    stats.columns = ['model', 'mean_ms', 'median_ms', 'std_ms', 'count']

    return stats


def compute_betting_efficiency(hands_df: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    """Compute betting efficiency metrics per model."""
    # Parse winners from hands
    def parse_winners(w):
        try:
            return json.loads(w) if w else []
        except:
            return []

    hands_df = hands_df.copy()
    hands_df['winners_list'] = hands_df['winners'].apply(parse_winners)

    action_df = action_df.copy()
    action_df['model'] = action_df['player_name'].apply(extract_model_name)

    results = []
    for model in action_df['model'].unique():
        model_wins = []

        for _, hand in hands_df.iterrows():
            winners = hand['winners_list']
            # Check if this model won (any player with this model name)
            model_won = any(extract_model_name(w) == model for w in winners)

            if model_won:
                # Get amount invested by this model in this hand
                hand_actions = action_df[
                    (action_df['game_id'] == hand['game_id']) &
                    (action_df['hand_number'] == hand['hand_number']) &
                    (action_df['model'] == model)
                ]
                invested = hand_actions['amount'].sum()
                model_wins.append({
                    'pot': hand['pot'],
                    'invested': invested,
                })

        if model_wins:
            df = pd.DataFrame(model_wins)
            # Calculate ROI only for hands where investment > 0
            df_with_investment = df[df['invested'] > 0]
            avg_roi = (df_with_investment['pot'] / df_with_investment['invested']).mean() if len(df_with_investment) > 0 else 0

            results.append({
                'model': model,
                'avg_pot_won': df['pot'].mean(),
                'total_pots_won': df['pot'].sum(),
                'hands_won': len(df),
                'avg_invested_per_win': df['invested'].mean(),
                'avg_roi_per_win': avg_roi,
            })

    return pd.DataFrame(results).sort_values('avg_roi_per_win', ascending=False) if results else pd.DataFrame()


def create_betting_efficiency_chart(efficiency_df: pd.DataFrame) -> str:
    """Create scatter plot of betting efficiency."""
    if len(efficiency_df) == 0:
        return "<p>No betting efficiency data available.</p>"

    fig = px.scatter(
        efficiency_df,
        x='avg_invested_per_win',
        y='avg_pot_won',
        size='hands_won',
        text='model',
        title='Betting Efficiency: Investment vs Return',
        labels={
            'avg_invested_per_win': 'Avg Chips Invested per Win',
            'avg_pot_won': 'Avg Pot Won',
        },
        color='avg_roi_per_win',
        color_continuous_scale='RdYlGn',
    )

    # Add diagonal line for 1:1 (break even)
    max_val = max(efficiency_df['avg_invested_per_win'].max(), efficiency_df['avg_pot_won'].max())
    fig.add_shape(
        type='line',
        x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color='gray', dash='dash'),
    )
    fig.add_annotation(x=max_val*0.7, y=max_val*0.6, text="1:1 (break even)",
                       showarrow=False, font=dict(size=10, color='gray'))

    fig.update_traces(textposition='top center')
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_latency_chart(latency_stats: pd.DataFrame) -> str:
    """Create latency comparison chart."""
    if len(latency_stats) == 0:
        return "<p>No latency data available.</p>"

    sorted_stats = latency_stats.sort_values('median_ms', ascending=True)
    colors = [MODEL_COLORS.get(m, '#888888') for m in sorted_stats['model']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_stats['median_ms'],
        y=sorted_stats['model'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:,.0f}ms" for v in sorted_stats['median_ms']],
        textposition='outside',
    ))

    fig.update_layout(
        title="Median Response Latency by Model",
        xaxis_title="Latency (ms)",
        yaxis_title="Model",
        template="plotly_white",
        height=400,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_html_report(
    rankings: pd.DataFrame,
    h2h: pd.DataFrame,
    action_stats: pd.DataFrame,
    efficiency: pd.DataFrame,
    total_hands: int,
    total_games: int,
    vpip_pfr: pd.DataFrame = None,
    profile_summary: pd.DataFrame = None,
    conf_stats: pd.DataFrame = None,
    latency_stats: pd.DataFrame = None,
    total_api_cost: float = 0,
    betting_efficiency: pd.DataFrame = None,
) -> str:
    """Generate the full HTML report."""

    # Generate all charts
    rankings_chart = create_rankings_chart(rankings)
    profit_chart = create_profit_chart(rankings)
    h2h_chart = create_h2h_heatmap(h2h)
    roi_chart = create_roi_chart(rankings)
    action_chart = create_action_distribution_chart(action_stats)
    cost_chart = create_cost_efficiency_chart(efficiency)
    profit_dollar_chart = create_profit_per_dollar_chart(efficiency)
    style_chart = create_playing_style_radar(action_stats, rankings)

    # New charts
    vpip_pfr_chart = create_vpip_pfr_chart(vpip_pfr) if vpip_pfr is not None and len(vpip_pfr) > 0 else ""
    profile_table = create_opponent_profile_table_html(profile_summary) if profile_summary is not None else ""
    conf_chart = create_confidence_chart(conf_stats) if conf_stats is not None else ""
    latency_chart = create_latency_chart(latency_stats) if latency_stats is not None else ""
    betting_eff_chart = create_betting_efficiency_chart(betting_efficiency) if betting_efficiency is not None and len(betting_efficiency) > 0 else ""

    # Top performers
    top_bb = rankings.iloc[0]
    top_roi = rankings.sort_values('avg_roi', ascending=False).iloc[0]

    if len(efficiency) > 0:
        eff_valid = efficiency[efficiency['profit_per_dollar'].notna()]
        top_efficient = eff_valid.sort_values('profit_per_dollar', ascending=False).iloc[0] if len(eff_valid) > 0 else None
    else:
        top_efficient = None

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Poker Arena - Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .highlight {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .highlight .stat-value {{
            color: white;
        }}
        .highlight .stat-label {{
            color: rgba(255,255,255,0.9);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 800px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
        }}
        dl {{
            margin: 0;
        }}
        dt {{
            color: #667eea;
            margin-top: 12px;
        }}
        dd {{
            margin-left: 0;
            margin-bottom: 8px;
            color: #555;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        h3 {{
            color: #444;
            font-size: 1.1em;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Poker Arena</h1>
        <p>Analysis Report - {datetime.now().strftime('%B %d, %Y')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card highlight">
                <div class="stat-value">{total_hands:,}</div>
                <div class="stat-label">Total Hands Played</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-value">{total_games}</div>
                <div class="stat-label">Games Completed</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-value">{len(rankings)}</div>
                <div class="stat-label">Models Tested</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-value">${total_api_cost:.2f}</div>
                <div class="stat-label">Total API Cost</div>
            </div>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{top_bb['model']}</div>
                <div class="stat-label">Best by BB/100 ({top_bb['avg_bb_100']:.2f})</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{top_roi['model']}</div>
                <div class="stat-label">Best ROI ({top_roi['avg_roi']*100:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{top_efficient['model'] if top_efficient is not None else 'N/A'}</div>
                <div class="stat-label">Most Cost-Efficient</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{rankings['total_rebuys'].sum()}</div>
                <div class="stat-label">Total Rebuys</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Glossary</h2>
        <p>Key poker and analysis terms used in this report:</p>
        <div class="two-col">
            <div>
                <h3>Performance Metrics</h3>
                <dl>
                    <dt><strong>BB/100</strong></dt>
                    <dd>Big Blinds won per 100 hands. The standard measure of poker win rate. +10 BB/100 means winning 10 big blinds every 100 hands on average.</dd>

                    <dt><strong>ROI (Return on Investment)</strong></dt>
                    <dd>Profit divided by total money invested. 50% ROI means you made back your investment plus 50% more.</dd>

                    <dt><strong>Win Rate</strong></dt>
                    <dd>Percentage of hands won (took the pot).</dd>

                    <dt><strong>Rebuys</strong></dt>
                    <dd>Number of times a player went broke and bought back in with a fresh stack.</dd>

                    <dt><strong>Profit</strong></dt>
                    <dd>Final chips minus total invested (starting stack + rebuys). Positive = won chips, negative = lost chips.</dd>
                </dl>
            </div>
            <div>
                <h3>Playing Style Stats</h3>
                <dl>
                    <dt><strong>VPIP (Voluntarily Put $ In Pot)</strong></dt>
                    <dd>% of hands where player voluntarily put money in preflop (called or raised). High VPIP = plays many hands (loose). Low VPIP = plays few hands (tight).</dd>

                    <dt><strong>PFR (Pre-Flop Raise)</strong></dt>
                    <dd>% of hands where player raised preflop. High PFR = aggressive. Low PFR = passive.</dd>

                    <dt><strong>Aggression Factor</strong></dt>
                    <dd>Ratio of aggressive actions (bets/raises) to passive actions (calls/checks). &gt;1 = aggressive, &lt;1 = passive.</dd>

                    <dt><strong>WTSD (Went To ShowDown)</strong></dt>
                    <dd>% of hands that reached showdown (cards revealed). High = calls down a lot, low = folds often.</dd>

                    <dt><strong>Fold to Raise</strong></dt>
                    <dd>% of time player folds when facing a raise. High = easily bluffed.</dd>
                </dl>
            </div>
        </div>
        <div class="two-col">
            <div>
                <h3>Playing Style Types</h3>
                <dl>
                    <dt><strong>TAG (Tight-Aggressive)</strong></dt>
                    <dd>Plays few hands but bets/raises aggressively. Generally the winning style.</dd>

                    <dt><strong>LAG (Loose-Aggressive)</strong></dt>
                    <dd>Plays many hands and bets/raises aggressively. High variance, can be profitable.</dd>

                    <dt><strong>Nit</strong></dt>
                    <dd>Plays very few hands, very tight. Easy to exploit by stealing blinds.</dd>

                    <dt><strong>Calling Station</strong></dt>
                    <dd>Calls too much, rarely raises or folds. Easy to value bet against, hard to bluff.</dd>
                </dl>
            </div>
            <div>
                <h3>Other Terms</h3>
                <dl>
                    <dt><strong>Showdown</strong></dt>
                    <dd>When remaining players reveal their cards at the end of a hand to determine the winner.</dd>

                    <dt><strong>C-Bet (Continuation Bet)</strong></dt>
                    <dd>Betting on the flop after raising preflop, "continuing" aggression.</dd>

                    <dt><strong>SPR (Stack-to-Pot Ratio)</strong></dt>
                    <dd>Effective stack divided by pot size. Low SPR = committed to pot, high SPR = more room to maneuver.</dd>

                    <dt><strong>Pot Odds</strong></dt>
                    <dd>Ratio of current pot to cost of calling. If pot is $100 and call is $20, pot odds are 5:1 (20%).</dd>

                    <dt><strong>ROI per Win (Betting Efficiency)</strong></dt>
                    <dd>Average pot won divided by average chips invested in winning hands. 3x ROI = winning 3 chips for every 1 invested. Higher = more efficient value extraction.</dd>
                </dl>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Model Rankings</h2>
        <p>Rankings based on BB/100 (big blinds won per 100 hands) - the standard poker performance metric.</p>
        <div class="two-col">
            <div class="chart-container">
                {rankings_chart}
            </div>
            <div class="chart-container">
                {profit_chart}
            </div>
        </div>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Total Profit</th>
                <th>BB/100</th>
                <th>Win Rate</th>
                <th>ROI</th>
                <th>Hands Played</th>
                <th>Rebuys</th>
            </tr>
            {"".join(f'''
            <tr>
                <td>{i+1}</td>
                <td><strong>{row['model']}</strong></td>
                <td style="color: {'#51cf66' if row['total_profit'] > 0 else '#ff6b6b'}">{row['total_profit']:,.0f}</td>
                <td>{row['avg_bb_100']:.2f}</td>
                <td>{row['win_rate']*100:.1f}%</td>
                <td>{row['avg_roi']*100:.1f}%</td>
                <td>{row['total_hands']:,}</td>
                <td>{row['total_rebuys']}</td>
            </tr>
            ''' for i, row in rankings.iterrows())}
        </table>
    </div>

    <div class="section">
        <h2>Head-to-Head Results</h2>
        <p>Percentage of hands won by each model (row) against each opponent (column), based on actual hands won. Green = winning more hands, Red = losing more hands.</p>
        <div class="chart-container">
            {h2h_chart}
        </div>
    </div>

    <div class="section">
        <h2>Playing Style Analysis</h2>
        <p>VPIP (Voluntarily Put $ In Pot) vs PFR (Pre-Flop Raise) - key metrics for classifying playing styles.</p>
        <div class="chart-container">
            {vpip_pfr_chart if vpip_pfr_chart else '<p>No preflop action data available.</p>'}
        </div>
    </div>

    <div class="section">
        <h2>Opponent Profiling</h2>
        <p>How each model is perceived by opponents. Aggregated statistics from opponent profile tracking.</p>
        {profile_table if profile_table else '<p>No opponent profile data available.</p>'}
    </div>

    <div class="section">
        <h2>Betting Efficiency</h2>
        <p>How efficiently do models extract value when they win? Higher ROI = winning bigger pots with smaller investments.</p>
        <div class="chart-container">
            {betting_eff_chart if betting_eff_chart else '<p>No betting efficiency data available.</p>'}
        </div>
        {f'''
        <table>
            <tr>
                <th>Model</th>
                <th>Avg Pot Won</th>
                <th>Avg Invested</th>
                <th>ROI per Win</th>
                <th>Hands Won</th>
            </tr>
            {"".join(f"""
            <tr>
                <td><strong>{row['model']}</strong></td>
                <td>{row['avg_pot_won']:.0f}</td>
                <td>{row['avg_invested_per_win']:.0f}</td>
                <td style="color: {'#51cf66' if row['avg_roi_per_win'] > 2 else '#ffa94d' if row['avg_roi_per_win'] > 1.5 else '#ff6b6b'}">{row['avg_roi_per_win']:.2f}x</td>
                <td>{row['hands_won']:,}</td>
            </tr>
            """ for _, row in betting_efficiency.iterrows())}
        </table>
        ''' if betting_efficiency is not None and len(betting_efficiency) > 0 else '<p>No betting efficiency data available.</p>'}
    </div>

    <div class="section">
        <h2>Decision Quality Analysis</h2>
        <div class="two-col">
            <div class="chart-container">
                {roi_chart}
            </div>
            <div class="chart-container">
                {action_chart}
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Cost-Efficiency Analysis</h2>
        <p>How much profit does each model generate relative to API costs?</p>
        <div class="two-col">
            <div class="chart-container">
                {profit_dollar_chart}
            </div>
            <div class="chart-container">
                {cost_chart}
            </div>
        </div>
        <table>
            <tr>
                <th>Model</th>
                <th>Total Profit</th>
                <th>API Cost</th>
                <th>Profit per $1</th>
                <th>Cost per Decision</th>
            </tr>
            {"".join(f'''
            <tr>
                <td><strong>{row['model']}</strong></td>
                <td>{row['profit']:,.0f}</td>
                <td>${row['estimated_cost']:.2f}</td>
                <td>{f"{row['profit_per_dollar']:.0f}" if pd.notna(row['profit_per_dollar']) else 'N/A'}</td>
                <td>{f"${row['cost_per_decision']:.4f}" if pd.notna(row['cost_per_decision']) else 'N/A'}</td>
            </tr>
            ''' for _, row in efficiency.sort_values('profit_per_dollar', ascending=False).iterrows())}
        </table>
    </div>

    <div class="section">
        <h2>Confidence Calibration</h2>
        <p>Do models with higher confidence actually win more? Points above the diagonal line indicate overconfidence.</p>
        <div class="chart-container">
            {conf_chart if conf_chart else '<p>No confidence data available.</p>'}
        </div>
    </div>

    <div class="section">
        <h2>Response Latency</h2>
        <p>Median decision-making time for each model. Faster isn't always better - some complex reasoning takes time.</p>
        <div class="chart-container">
            {latency_chart if latency_chart else '<p>No latency data available.</p>'}
        </div>
    </div>

    <div class="section">
        <h2>Behavioral Profiling</h2>
        <p>Playing style analysis showing fold/check/call/raise tendencies and win rates.</p>
        <div class="chart-container">
            {style_chart}
        </div>
    </div>

    <div class="footer">
        <p>Generated by LLM Poker Arena Analysis Dashboard</p>
        <p>Project by Matthew Ohanian | {datetime.now().strftime('%Y')}</p>
    </div>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate LLM Poker analysis report")
    parser.add_argument(
        "--output", "-o",
        default="analysis_report.html",
        help="Output HTML file path"
    )
    args = parser.parse_args()

    print("Loading data from database...")

    # Check database exists
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Run some experiments first!")
        return

    # Load all data
    game_df = load_game_data()

    if len(game_df) == 0:
        print("No completed games found in database.")
        print("Run some experiments first!")
        return

    action_df = load_action_data()
    showdown_df = load_showdown_data()
    cost_df = load_cost_data()
    profiles_df = load_opponent_profiles()
    hands_df = load_hands_data()

    print(f"Found {len(game_df)} player results across games")

    # Compute metrics
    print("Computing rankings...")
    rankings = compute_model_rankings(game_df)

    print("Computing head-to-head results...")
    h2h = compute_head_to_head(game_df)

    print("Computing action statistics...")
    action_stats = compute_action_stats(action_df)

    print("Computing cost efficiency...")
    efficiency = compute_cost_efficiency(game_df, cost_df)

    print("Computing VPIP/PFR stats...")
    vpip_pfr = compute_vpip_pfr(action_df)

    print("Computing opponent profile summary...")
    profile_summary = compute_opponent_profile_summary(profiles_df)

    print("Computing confidence calibration...")
    conf_stats = compute_confidence_stats(action_df, hands_df)

    print("Computing latency stats...")
    latency_stats = compute_latency_stats(action_df)

    print("Computing betting efficiency...")
    betting_efficiency = compute_betting_efficiency(hands_df, action_df)

    # Summary stats (deduplicate by game_id to avoid double-counting hands)
    total_hands = game_df.drop_duplicates('game_id')['num_hands'].sum()
    total_games = game_df['game_id'].nunique()
    total_api_cost = cost_df['estimated_cost'].sum() if len(cost_df) > 0 else 0

    # Generate report
    print("Generating HTML report...")
    html = generate_html_report(
        rankings=rankings,
        h2h=h2h,
        action_stats=action_stats,
        efficiency=efficiency,
        total_hands=total_hands,
        total_games=total_games,
        vpip_pfr=vpip_pfr,
        profile_summary=profile_summary,
        conf_stats=conf_stats,
        latency_stats=latency_stats,
        total_api_cost=total_api_cost,
        betting_efficiency=betting_efficiency,
    )

    # Write file
    output_path = Path(args.output)
    output_path.write_text(html)

    print(f"\nReport generated: {output_path.absolute()}")
    print(f"Open in browser to view results!")


if __name__ == "__main__":
    main()
