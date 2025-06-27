import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import sys
from PIL import Image, ImageTk

# Add the src directory to the path so we can import from judgement_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Import the game components
from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent


class CardImageManager:
    """Manages card images for the GUI."""

    def __init__(self):
        self.card_images = {}
        self.card_back_image = None
        self.load_card_images()

    def load_card_images(self):
        """Load or create card images."""
        # Create a simple card image generator since we don't have actual card images
        self.create_simple_card_images()

    def create_simple_card_images(self):
        """Create simple colored rectangles for cards."""
        # Card dimensions - made bigger
        card_width, card_height = 100, 140

        # Colors for suits
        suit_colors = {
            "Hearts": "#ff0000",  # Red
            "Diamonds": "#ff0000",  # Red
            "Clubs": "#000000",  # Black
            "Spades": "#000000",  # Black
        }

        # Create card back image
        self.card_back_image = self.create_card_image(
            "BACK", "#0066cc", card_width, card_height
        )

        # Create images for each card
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

        for suit in suits:
            for rank in ranks:
                card_name = f"{rank} of {suit}"
                color = suit_colors[suit]
                self.card_images[card_name] = self.create_card_image(
                    rank, color, card_width, card_height, suit
                )

    def create_card_image(self, text, color, width, height, suit=None):
        """Create a simple card image with text."""
        # Create image
        img = Image.new("RGB", (width, height), "white")

        # Create a drawing context
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=3)

        # Try to use a nice font, fallback to default - made bigger
        try:
            font_size = 24 if len(text) <= 2 else 20  # Bigger font
            font = ImageFont.truetype("Arial", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2 - 15  # Moved up a bit

        # Draw text
        draw.text((x, y), text, fill=color, font=font)

        # Add suit symbol if provided - made bigger
        if suit:
            suit_symbols = {"Hearts": "♥", "Diamonds": "♦", "Clubs": "♣", "Spades": "♠"}
            if suit in suit_symbols:
                suit_text = suit_symbols[suit]
                try:
                    suit_font = ImageFont.truetype("Arial", 32)  # Bigger suit symbol
                except:
                    suit_font = ImageFont.load_default()

                suit_bbox = draw.textbbox((0, 0), suit_text, font=suit_font)
                suit_width = suit_bbox[2] - suit_bbox[0]
                suit_height = suit_bbox[3] - suit_bbox[1]

                suit_x = (width - suit_width) // 2
                suit_y = y + text_height + 10

                draw.text((suit_x, suit_y), suit_text, fill=color, font=suit_font)

        return ImageTk.PhotoImage(img)

    def get_card_image(self, card_name):
        """Get the image for a card."""
        return self.card_images.get(card_name, self.card_back_image)

    def get_card_back_image(self):
        """Get the card back image."""
        return self.card_back_image


class JudgementGUI:
    """
    GUI interface for playing against the trained Judgement AI agent.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Judgement AI - Play Against AI")
        self.root.geometry("1200x800")  # Made smaller since we removed right sidebar

        # Game components
        self.env = None
        self.state_encoder = None
        self.agent = None
        self.current_state = None
        self.game_phase = "setup"  # setup, bidding, playing, game_over

        # Card image manager
        self.card_manager = CardImageManager()

        # Trick completion tracking
        self.last_trick_count = 0
        self.trick_complete_delay = 5000  # 5 seconds in milliseconds
        self.trick_complete_pending = (
            False  # Flag to track if we're waiting for trick completion
        )
        self.complete_trick_for_display = (
            []
        )  # Store complete trick for display during delay
        self.pending_action = None  # Store pending action for delayed trick resolution
        self.pending_player = None  # Store which player the pending action is for
        self.pending_is_human = False  # Store if the pending action is from human

        # Button state tracking
        self.card_buttons = []  # Store references to card buttons
        self.bid_buttons = []  # Store references to bid buttons

        # GUI components
        self.setup_gui()

        # Initialize game
        self.initialize_game()

    def setup_gui(self):
        """Set up the GUI layout and components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title - made bigger
        title_label = ttk.Label(
            main_frame,
            text="Judgement AI - Play Against AI",
            font=("Arial", 20, "bold"),  # Bigger font
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Left panel - Game info and controls
        left_panel = ttk.LabelFrame(main_frame, text="Game Information", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Game status - made bigger
        self.status_label = ttk.Label(
            left_panel, text="Loading...", font=("Arial", 14)
        )  # Bigger font
        self.status_label.grid(row=0, column=0, pady=(0, 10))

        # Round info - made bigger
        self.round_label = ttk.Label(
            left_panel, text="Round: -", font=("Arial", 12)
        )  # Bigger font
        self.round_label.grid(row=1, column=0, pady=(0, 5))

        self.trump_label = ttk.Label(
            left_panel, text="Trump: -", font=("Arial", 12)
        )  # Bigger font
        self.trump_label.grid(row=2, column=0, pady=(0, 5))

        self.cards_label = ttk.Label(
            left_panel, text="Cards: -", font=("Arial", 12)
        )  # Bigger font
        self.cards_label.grid(row=3, column=0, pady=(0, 10))

        # Bidding info - made bigger
        self.bidding_frame = ttk.LabelFrame(left_panel, text="Bidding", padding="5")
        self.bidding_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.bid_labels = []
        for i in range(4):
            player_name = "You" if i == 0 else f"AI {i}"
            label = ttk.Label(
                self.bidding_frame, text=f"{player_name}: -", font=("Arial", 11)
            )  # Bigger font
            label.grid(row=i, column=0, sticky=tk.W)
            self.bid_labels.append(label)

        # Tricks won info - made bigger
        self.tricks_frame = ttk.LabelFrame(left_panel, text="Tricks Won", padding="5")
        self.tricks_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.trick_labels = []
        for i in range(4):
            player_name = "You" if i == 0 else f"AI {i}"
            label = ttk.Label(
                self.tricks_frame, text=f"{player_name}: 0", font=("Arial", 11)
            )  # Bigger font
            label.grid(row=i, column=0, sticky=tk.W)
            self.trick_labels.append(label)

        # Current trick - now with card images
        self.current_trick_frame = ttk.LabelFrame(
            left_panel, text="Current Trick", padding="5"
        )
        self.current_trick_frame.grid(
            row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.current_trick_canvas = tk.Canvas(
            self.current_trick_frame,
            width=350,  # Made wider
            height=180,  # Made taller
            bg="white",
            relief="sunken",
            bd=1,
        )
        self.current_trick_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Control buttons - made bigger
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.new_game_btn = ttk.Button(
            button_frame, text="New Game", command=self.new_game
        )
        self.new_game_btn.grid(row=0, column=0, padx=(0, 5))

        self.load_model_btn = ttk.Button(
            button_frame, text="Load Model", command=self.load_model
        )
        self.load_model_btn.grid(row=0, column=1, padx=(0, 5))

        # Center panel - Player's hand
        center_panel = ttk.LabelFrame(main_frame, text="Your Hand", padding="10")
        center_panel.grid(
            row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10)
        )
        center_panel.columnconfigure(0, weight=1)

        self.hand_frame = ttk.Frame(center_panel)
        self.hand_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Action panel
        action_panel = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_panel.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(10, 0))

        self.action_frame = ttk.Frame(action_panel)
        self.action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Configure grid weights for resizing
        main_frame.rowconfigure(1, weight=1)
        center_panel.rowconfigure(0, weight=1)

    def initialize_game(self):
        """Initialize the game environment and components."""
        try:
            # Initialize environment
            self.env = JudgementEnv(num_players=4, max_cards=7)
            self.state_encoder = StateEncoder(num_players=4, max_cards=7)

            # Try to load the best agent
            model_path = "models/selfplay_best_agent.pth"
            if os.path.exists(model_path):
                self.load_agent(model_path)
                self.status_label.config(text="Ready to play! Loaded trained agent.")
            else:
                self.status_label.config(
                    text="No trained model found. Please load a model."
                )

            # Start new game
            self.new_game()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize game: {str(e)}")
            self.status_label.config(text="Error initializing game")

    def load_agent(self, model_path: str):
        """Load a trained agent from file."""
        try:
            self.agent = PPOAgent(self.state_encoder)
            self.agent.load_model(model_path)
            print(f"Loaded agent from {model_path}")
        except Exception as e:
            print(f"Failed to load agent: {e}")
            self.agent = None

    def load_model(self):
        """Open file dialog to load a model."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")],
        )
        if file_path:
            self.load_agent(file_path)
            self.status_label.config(
                text=f"Loaded model: {os.path.basename(file_path)}"
            )

    def new_game(self):
        """Start a new game."""
        if self.env is None:
            return

        # Reset environment
        self.current_state = self.env.reset()
        self.game_phase = "bidding"

        # Reset trick completion tracking
        self.last_trick_count = 0
        self.trick_complete_pending = False
        self.complete_trick_for_display = []  # Clear stored trick
        self.pending_action = None  # Store pending action for delayed trick resolution
        self.pending_player = None  # Store which player the pending action is for
        self.pending_is_human = False  # Store if the pending action is from human

        # Clear button references
        self.card_buttons.clear()
        self.bid_buttons.clear()

        # Update display
        self.update_display()

        # Start the game loop
        self.game_loop()

    def update_display(self):
        """Update the GUI display with current game state."""
        if self.current_state is None:
            return

        # Update round info
        self.round_label.config(text=f"Round: {self.env.round_number}")
        self.trump_label.config(text=f"Trump: {self.env.trump}")
        self.cards_label.config(text=f"Cards: {self.env.round_cards}")

        # Update bidding info
        for i, bid in enumerate(self.env.declarations):
            player_name = "You" if i == 0 else f"AI {i}"
            if bid is not None:
                self.bid_labels[i].config(text=f"{player_name}: {bid}")
            else:
                self.bid_labels[i].config(text=f"{player_name}: -")

        # Update tricks won
        for i, tricks in enumerate(self.env.tricks_won):
            player_name = "You" if i == 0 else f"AI {i}"
            self.trick_labels[i].config(text=f"{player_name}: {tricks}")

        # Update current trick with card images
        self.update_current_trick_display()

        # Update status
        if self.env.phase == "bidding":
            if self.env.current_player == 0:
                self.status_label.config(text="Your turn to bid!")
            else:
                self.status_label.config(
                    text=f"AI Player {self.env.current_player} is bidding..."
                )
        else:
            if self.env.current_player == 0:
                self.status_label.config(text="Your turn to play a card!")
            else:
                self.status_label.config(
                    text=f"AI Player {self.env.current_player} is playing..."
                )

        # Update button states
        self.update_button_states()

    def update_button_states(self):
        """Update the state of buttons based on current game phase and player turn."""
        is_human_turn = self.env.current_player == 0
        is_bidding_phase = self.env.phase == "bidding"
        is_playing_phase = self.env.phase == "playing"
        is_waiting_for_trick = self.trick_complete_pending

        # Disable all buttons if waiting for trick completion
        if is_waiting_for_trick:
            self._disable_all_buttons()
            return

        # During bidding phase
        if is_bidding_phase:
            if is_human_turn:
                # Enable bid buttons, disable card buttons
                self._enable_bid_buttons()
                self._disable_card_buttons()
            else:
                # Disable all buttons during AI bidding
                self._disable_all_buttons()

        # During playing phase
        elif is_playing_phase:
            if is_human_turn:
                # Enable card buttons, disable bid buttons
                self._enable_card_buttons()
                self._disable_bid_buttons()
            else:
                # Disable all buttons during AI playing
                self._disable_all_buttons()

    def _disable_all_buttons(self):
        """Disable all interactive buttons."""
        self._disable_card_buttons()
        self._disable_bid_buttons()

    def _disable_card_buttons(self):
        """Disable all card buttons."""
        for btn in self.card_buttons:
            btn.config(state="disabled")

    def _enable_card_buttons(self):
        """Enable all card buttons."""
        for btn in self.card_buttons:
            btn.config(state="normal")

    def _disable_bid_buttons(self):
        """Disable all bid buttons."""
        for btn in self.bid_buttons:
            btn.config(state="disabled")

    def _enable_bid_buttons(self):
        """Enable all bid buttons."""
        for btn in self.bid_buttons:
            btn.config(state="normal")

    def update_current_trick_display(self):
        """Update the current trick display with card images."""
        # Clear the canvas
        self.current_trick_canvas.delete("all")

        # Use stored complete trick if we're in delay period, otherwise use current trick
        if self.trick_complete_pending and self.complete_trick_for_display:
            trick_to_display = self.complete_trick_for_display
        else:
            trick_to_display = (
                self.env.current_trick if hasattr(self.env, "current_trick") else []
            )

        if not trick_to_display:
            # Show "No cards played yet" text
            self.current_trick_canvas.create_text(
                175,
                90,  # Centered
                text="No cards played yet",
                font=("Arial", 14),  # Bigger font
                fill="gray",
            )
            return

        # Display cards in the current trick
        card_width = 70  # Made bigger
        card_height = 100  # Made bigger
        spacing = 15  # More spacing
        start_x = 20

        for i, (player, card) in enumerate(trick_to_display):
            x = start_x + i * (card_width + spacing)
            y = 40

            # Get card image
            card_image = self.card_manager.get_card_image(card)

            # Create card image on canvas
            image_id = self.current_trick_canvas.create_image(
                x, y, image=card_image, anchor="nw"
            )

            # Add player label
            player_name = "You" if player == 0 else f"AI {player}"
            self.current_trick_canvas.create_text(
                x + card_width // 2,
                y + card_height + 15,  # More spacing
                text=player_name,
                font=("Arial", 12),  # Bigger font
                fill="black",
            )

    def display_hand(self):
        """Display the player's hand as clickable card images."""
        # Clear existing hand and button references
        for widget in self.hand_frame.winfo_children():
            widget.destroy()
        self.card_buttons.clear()

        if self.env is None:
            return

        # Always show player 0's hand (the human player)
        hand = self.env.hands[0]

        # Create card buttons in a grid
        cards_per_row = 3  # Reduced to 3 per row for bigger cards
        for i, card in enumerate(hand):
            row = i // cards_per_row
            col = i % cards_per_row

            # Create frame for each card
            card_frame = ttk.Frame(self.hand_frame)
            card_frame.grid(row=row, column=col, padx=8, pady=8)  # More padding

            # Get card image
            card_image = self.card_manager.get_card_image(card)

            # Create button with card image
            card_btn = tk.Button(
                card_frame,
                image=card_image,
                command=lambda idx=i: self.play_card(idx),
                relief="raised",
                bd=3,  # Bigger border
            )
            card_btn.pack()

            # Keep reference to prevent garbage collection
            card_btn.image = card_image

            # Store reference to button for state management
            self.card_buttons.append(card_btn)

        # Configure grid weights
        for i in range((len(hand) + cards_per_row - 1) // cards_per_row):
            self.hand_frame.rowconfigure(i, weight=1)
        for i in range(cards_per_row):
            self.hand_frame.columnconfigure(i, weight=1)

        # Update button states after creating buttons
        self.update_button_states()

    def display_bidding_buttons(self):
        """Display bidding buttons."""
        # Clear existing action buttons and button references
        for widget in self.action_frame.winfo_children():
            widget.destroy()
        self.bid_buttons.clear()

        # Create bidding buttons - made bigger
        ttk.Label(
            self.action_frame, text="Select your bid:", font=("Arial", 12)
        ).grid(  # Bigger font
            row=0, column=0, columnspan=3, pady=(0, 10)  # More padding
        )

        for i in range(self.env.round_cards + 1):
            bid_btn = ttk.Button(
                self.action_frame, text=str(i), command=lambda bid=i: self.make_bid(bid)
            )
            bid_btn.grid(row=1, column=i, padx=4, pady=4)  # More padding

            # Store reference to button for state management
            self.bid_buttons.append(bid_btn)

        # Update button states after creating buttons
        self.update_button_states()

    def make_bid(self, bid: int):
        """Make a bid."""
        if self.env.current_player != 0:
            return

        # Validate bid
        if not self._is_valid_bid(bid):
            messagebox.showwarning(
                "Invalid Bid",
                "This bid would make the total equal to the number of players!",
            )
            return

        # Make the bid
        self._take_action(bid)

    def _is_valid_bid(self, bid: int) -> bool:
        """Check if a bid is valid."""
        # Count existing bids
        num_bids = sum(1 for d in self.env.declarations if d is not None)
        if num_bids == self.env.num_players - 1:
            current_total = sum(d for d in self.env.declarations if d is not None)
            return current_total + bid != self.env.num_players
        return True

    def play_card(self, card_idx: int):
        """Play a card."""
        if self.env.current_player != 0:
            return

        # Validate card
        if not self._is_legal_card(card_idx):
            messagebox.showwarning("Invalid Card", "You must follow suit if possible!")
            return

        # Play the card
        self._take_action(card_idx)

    def _is_legal_card(self, card_idx: int) -> bool:
        """Check if a card play is legal."""
        if self.env is None or card_idx >= len(self.env.hands[0]):
            return False

        card = self.env.hands[0][card_idx]
        return self.env._is_legal_card(0, card)

    def _take_action(self, action: int):
        """Take an action (bid or play card)."""
        # Only applies to human (player 0)
        current_trick_before = (
            self.env.current_trick.copy() if hasattr(self.env, "current_trick") else []
        )
        is_playing_phase = self.env.phase == "playing"
        is_4th_card = is_playing_phase and len(current_trick_before) == 3

        if is_4th_card:
            # Get the card that will be played
            card = self.env.hands[0][action]
            complete_trick = current_trick_before + [(0, card)]
            self.complete_trick_for_display = complete_trick
            self.trick_complete_pending = True
            self.pending_action = action
            self.pending_player = 0
            self.pending_is_human = True
            self.update_display()
            self.display_hand()
            self.status_label.config(text="Trick complete! Waiting 5 seconds...")
            self.root.after(self.trick_complete_delay, self._resolve_pending_trick)
            return

        # Otherwise, proceed as normal
        next_state, reward, done = self.env.step(0, action)
        self.current_state = next_state
        self.update_display()
        self.display_hand()
        if done:
            self.game_over()
            return
        self.last_trick_count = (
            len(self.env.current_trick) if hasattr(self.env, "current_trick") else 0
        )
        self.root.after(100, self.game_loop)

    def _resolve_pending_trick(self):
        """Resolve the trick after the delay for both human and AI."""
        if self.pending_action is not None and self.pending_player is not None:
            next_state, reward, done = self.env.step(
                self.pending_player, self.pending_action
            )
            self.current_state = next_state
            self.trick_complete_pending = False
            self.complete_trick_for_display = []
            self.pending_action = None
            self.pending_player = None
            self.pending_is_human = False
            self.update_display()
            if done:
                self.game_over()
                return
            self.last_trick_count = (
                len(self.env.current_trick) if hasattr(self.env, "current_trick") else 0
            )
            self.root.after(100, self.game_loop)

    def game_loop(self):
        """Main game loop for AI turns."""
        if self.current_state is None or self.env.current_player == 0:
            # Player's turn - update hand display
            if self.env.phase == "bidding":
                self.display_bidding_buttons()
            # Always display hand, even during bidding
            self.display_hand()
            # Update button states for player's turn
            self.update_button_states()
            return

        # If we're waiting for trick completion, don't continue
        if self.trick_complete_pending:
            return

        # AI's turn - disable all buttons
        self.update_button_states()

        if self.agent is None:
            # Random agent fallback
            legal_actions = self.env.get_legal_actions(self.env.current_player)
            action = np.random.choice(legal_actions)
        else:
            # Use trained agent
            action, _, _ = self.agent.select_action(
                self.current_state,
                self.env.get_legal_actions(self.env.current_player),
                epsilon=0.0,  # No exploration during play
            )

        current_trick_before = (
            self.env.current_trick.copy() if hasattr(self.env, "current_trick") else []
        )
        is_playing_phase = self.env.phase == "playing"
        is_4th_card = is_playing_phase and len(current_trick_before) == 3

        if is_4th_card:
            card = self.env.hands[self.env.current_player][action]
            complete_trick = current_trick_before + [(self.env.current_player, card)]
            self.complete_trick_for_display = complete_trick
            self.trick_complete_pending = True
            self.pending_action = action
            self.pending_player = self.env.current_player
            self.pending_is_human = False
            self.update_display()
            self.status_label.config(text="Trick complete! Waiting 5 seconds...")
            self.root.after(self.trick_complete_delay, self._resolve_pending_trick)
            return

        # Otherwise, proceed as normal
        next_state, reward, done = self.env.step(self.env.current_player, action)
        self.current_state = next_state
        self.update_display()
        if done:
            self.game_over()
            return
        self.last_trick_count = (
            len(self.env.current_trick) if hasattr(self.env, "current_trick") else 0
        )
        self.root.after(500, self.game_loop)  # Small delay for AI moves

    def game_over(self):
        """Handle game over."""
        self.game_phase = "game_over"

        # Calculate final scores
        scores = []
        for i in range(self.env.num_players):
            n = self.env.declarations[i]
            actual = self.env.tricks_won[i]
            if n == actual:
                score = 11 * n + 10
            else:
                score = -10 * abs(n - actual)
            scores.append(score)

        # Show results
        result_text = "Game Over!\n\nFinal Scores:\n"
        for i, score in enumerate(scores):
            player_name = "You" if i == 0 else f"AI Player {i}"
            result_text += f"{player_name}: {score}\n"

        messagebox.showinfo("Game Over", result_text)
        self.status_label.config(text="Game Over - Click 'New Game' to play again")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = JudgementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
