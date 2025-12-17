"""SQLite database for storing game data."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class GameDatabase:
    """SQLite database for poker game data."""

    def __init__(self, db_path: str = "data/games.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate_schema()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Game sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                num_players INTEGER,
                num_hands INTEGER,
                starting_stack INTEGER,
                small_blind INTEGER,
                big_blind INTEGER,
                players_config TEXT,
                status TEXT DEFAULT 'in_progress'
            )
        """)

        # Players in each game
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                player_name TEXT,
                model_id TEXT,
                starting_chips INTEGER,
                final_chips INTEGER,
                hands_won INTEGER DEFAULT 0,
                rebuys INTEGER DEFAULT 0,
                total_profit INTEGER,
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        """)

        # Individual hands
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                hand_number INTEGER,
                dealer TEXT,
                pot INTEGER,
                winners TEXT,
                board TEXT,
                went_to_showdown BOOLEAN,
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        """)

        # Player decisions/actions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                hand_id INTEGER,
                hand_number INTEGER,
                player_name TEXT,
                model_id TEXT,
                betting_round TEXT,
                action_type TEXT,
                amount INTEGER,
                reasoning TEXT,
                hole_cards TEXT,
                board TEXT,
                pot_before INTEGER,
                FOREIGN KEY (game_id) REFERENCES games(id),
                FOREIGN KEY (hand_id) REFERENCES hands(id)
            )
        """)

        # Showdown results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS showdowns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                hand_id INTEGER,
                hand_number INTEGER,
                player_name TEXT,
                hole_cards TEXT,
                hand_name TEXT,
                amount_won INTEGER,
                FOREIGN KEY (game_id) REFERENCES games(id),
                FOREIGN KEY (hand_id) REFERENCES hands(id)
            )
        """)

        # API usage/costs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                model_id TEXT,
                total_calls INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                estimated_cost REAL,
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        """)

        # Opponent profiles - what each agent thinks about others
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS opponent_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                observer_player TEXT,
                observer_model TEXT,
                observed_player TEXT,
                observed_model TEXT,
                hands_played INTEGER,
                vpip REAL,
                pfr REAL,
                aggression_factor REAL,
                cbet_frequency REAL,
                fold_to_raise_rate REAL,
                fold_to_cbet_rate REAL,
                wtsd REAL,
                avg_bet_sizing REAL,
                estimated_style TEXT,
                notes TEXT,
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        """)

        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Add new columns to tables if they don't exist."""
        cursor = self.conn.cursor()

        # Migrate actions table
        cursor.execute("PRAGMA table_info(actions)")
        existing = {row[1] for row in cursor.fetchall()}

        new_cols = [
            ("confidence", "REAL"),
            ("position", "TEXT"),
            ("opponent_read", "TEXT"),
            ("latency_ms", "INTEGER"),
            ("call_amount", "INTEGER"),
            ("pot_size", "INTEGER"),
            ("hand_strength", "REAL"),
            ("hand_name", "TEXT"),
            ("pot_odds", "REAL"),
            ("spr", "REAL"),
            ("effective_stack", "INTEGER"),
        ]

        for name, typ in new_cols:
            if name not in existing:
                cursor.execute(f"ALTER TABLE actions ADD COLUMN {name} {typ}")

        # Migrate game_players table
        cursor.execute("PRAGMA table_info(game_players)")
        existing_gp = {row[1] for row in cursor.fetchall()}

        if "rebuys" not in existing_gp:
            cursor.execute("ALTER TABLE game_players ADD COLUMN rebuys INTEGER DEFAULT 0")

        self.conn.commit()

    def start_game(
        self,
        num_players: int,
        num_hands: int,
        starting_stack: int,
        small_blind: int,
        big_blind: int,
        players_config: list[dict],
    ) -> int:
        """Start a new game session and return game_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO games (num_players, num_hands, starting_stack, small_blind, big_blind, players_config)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (num_players, num_hands, starting_stack, small_blind, big_blind, json.dumps(players_config)),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_player(
        self,
        game_id: int,
        player_name: str,
        model_id: str,
        starting_chips: int,
    ) -> None:
        """Add a player to a game."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO game_players (game_id, player_name, model_id, starting_chips)
            VALUES (?, ?, ?, ?)
            """,
            (game_id, player_name, model_id, starting_chips),
        )
        self.conn.commit()

    def start_hand(
        self,
        game_id: int,
        hand_number: int,
        dealer: str,
    ) -> int:
        """Start a new hand and return hand_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO hands (game_id, hand_number, dealer)
            VALUES (?, ?, ?)
            """,
            (game_id, hand_number, dealer),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_action(
        self,
        game_id: int,
        hand_id: int,
        hand_number: int,
        player_name: str,
        model_id: str,
        betting_round: str,
        action_type: str,
        amount: int,
        reasoning: str,
        hole_cards: str = "",
        board: str = "",
        pot_before: int = 0,
        # New metrics
        confidence: float | None = None,
        position: str | None = None,
        opponent_read: str | None = None,
        latency_ms: int | None = None,
        call_amount: int | None = None,
        pot_size: int | None = None,
        hand_strength: float | None = None,
        hand_name: str | None = None,
        pot_odds: float | None = None,
        spr: float | None = None,
        effective_stack: int | None = None,
    ) -> None:
        """Log a player action."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO actions (game_id, hand_id, hand_number, player_name, model_id,
                                betting_round, action_type, amount, reasoning, hole_cards, board, pot_before,
                                confidence, position, opponent_read, latency_ms, call_amount, pot_size,
                                hand_strength, hand_name, pot_odds, spr, effective_stack)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, hand_id, hand_number, player_name, model_id, betting_round,
             action_type, amount, reasoning, hole_cards, board, pot_before,
             confidence, position, opponent_read, latency_ms, call_amount, pot_size,
             hand_strength, hand_name, pot_odds, spr, effective_stack),
        )
        self.conn.commit()

    def end_hand(
        self,
        hand_id: int,
        pot: int,
        winners: list[str],
        board: str,
        went_to_showdown: bool,
    ) -> None:
        """Record hand result."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE hands SET pot = ?, winners = ?, board = ?, went_to_showdown = ?
            WHERE id = ?
            """,
            (pot, json.dumps(winners), board, went_to_showdown, hand_id),
        )
        self.conn.commit()

    def log_showdown(
        self,
        game_id: int,
        hand_id: int,
        hand_number: int,
        player_name: str,
        hole_cards: str,
        hand_name: str,
        amount_won: int,
    ) -> None:
        """Log showdown result for a player."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO showdowns (game_id, hand_id, hand_number, player_name, hole_cards, hand_name, amount_won)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, hand_id, hand_number, player_name, hole_cards, hand_name, amount_won),
        )
        self.conn.commit()

    def end_game(
        self,
        game_id: int,
        final_chips: dict[str, int],
        hands_won: dict[str, int],
        rebuys: dict[str, int],
        starting_stack: int,
        api_usage: dict[str, dict],
    ) -> None:
        """End a game and record final results."""
        cursor = self.conn.cursor()

        # Update game status
        cursor.execute(
            """
            UPDATE games SET ended_at = CURRENT_TIMESTAMP, status = 'completed'
            WHERE id = ?
            """,
            (game_id,),
        )

        # Update player final results
        # true_profit = final_chips - starting_chips - (rebuys * starting_stack)
        for player_name, chips in final_chips.items():
            player_rebuys = rebuys.get(player_name, 0)
            true_profit = chips - starting_stack - (player_rebuys * starting_stack)
            cursor.execute(
                """
                UPDATE game_players
                SET final_chips = ?, hands_won = ?, rebuys = ?, total_profit = ?
                WHERE game_id = ? AND player_name = ?
                """,
                (chips, hands_won.get(player_name, 0), player_rebuys, true_profit, game_id, player_name),
            )

        # Log API usage
        for model_id, usage in api_usage.items():
            cursor.execute(
                """
                INSERT INTO api_usage (game_id, model_id, total_calls, input_tokens, output_tokens, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (game_id, model_id, usage.get("calls", 0), usage.get("input_tokens", 0),
                 usage.get("output_tokens", 0), usage.get("cost", 0)),
            )

        self.conn.commit()

    def save_opponent_profile(
        self,
        game_id: int,
        observer_player: str,
        observer_model: str,
        observed_player: str,
        observed_model: str,
        profile: dict,
    ) -> None:
        """Save an opponent profile that an agent built."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO opponent_profiles (
                game_id, observer_player, observer_model, observed_player, observed_model,
                hands_played, vpip, pfr, aggression_factor, cbet_frequency,
                fold_to_raise_rate, fold_to_cbet_rate, wtsd, avg_bet_sizing,
                estimated_style, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                observer_player,
                observer_model,
                observed_player,
                observed_model,
                profile.get("hands_played", 0),
                profile.get("vpip", 0),
                profile.get("pfr", 0),
                profile.get("aggression_factor", 0),
                profile.get("cbet_frequency", 0),
                profile.get("fold_to_raise_rate", 0),
                profile.get("fold_to_cbet_rate", 0),
                profile.get("wtsd", 0),
                profile.get("avg_bet_sizing", 0),
                profile.get("estimated_style", "unknown"),
                json.dumps(profile.get("notes", [])),
            ),
        )
        self.conn.commit()

    def get_game_summary(self, game_id: int) -> dict:
        """Get summary of a game."""
        cursor = self.conn.cursor()

        # Get game info
        cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
        game = dict(cursor.fetchone())

        # Get players
        cursor.execute("SELECT * FROM game_players WHERE game_id = ?", (game_id,))
        players = [dict(row) for row in cursor.fetchall()]

        # Get hand count
        cursor.execute("SELECT COUNT(*) as count FROM hands WHERE game_id = ?", (game_id,))
        hands_played = cursor.fetchone()["count"]

        return {
            "game": game,
            "players": players,
            "hands_played": hands_played,
        }

    def get_model_stats(self) -> list[dict]:
        """Get aggregate stats by model across all games."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                model_id,
                COUNT(*) as games_played,
                SUM(hands_won) as total_hands_won,
                SUM(total_profit) as total_profit,
                AVG(total_profit) as avg_profit_per_game,
                SUM(final_chips - starting_chips) as net_chips
            FROM game_players
            GROUP BY model_id
            ORDER BY total_profit DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_games(self, limit: int = 10) -> list[dict]:
        """Get recent games."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT g.*,
                   GROUP_CONCAT(gp.player_name || ':' || gp.model_id) as players
            FROM games g
            LEFT JOIN game_players gp ON g.id = gp.game_id
            GROUP BY g.id
            ORDER BY g.started_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


# Global database instance
_db: GameDatabase | None = None


def get_database(db_path: str = "data/games.db") -> GameDatabase:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = GameDatabase(db_path)
    return _db


def reset_database() -> None:
    """Reset the global database instance."""
    global _db
    if _db is not None:
        _db.close()
    _db = None
