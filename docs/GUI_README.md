# Judgement AI GUI Interface

This GUI allows you to play against the trained Judgement AI agent in a user-friendly interface.

## Quick Start

1. **Make sure you have a trained model:**

   - The GUI will automatically try to load `models/selfplay_best_agent.pth`
   - If no model is found, you can load one using the "Load Model" button

2. **Run the GUI:**
   ```bash
   python play_against_ai.py
   ```
   or
   ```bash
   python gui_interface.py
   ```

## How to Play

### Game Setup

- You are **Player 0** (the human player)
- AI Players 1, 2, and 3 are controlled by the trained agent
- The game follows standard Judgement rules

### Bidding Phase

- Each round starts with bidding
- You must bid on how many tricks you think you'll win
- **Important Rule:** The total of all bids cannot equal the number of players (4)
- Click on a number button to make your bid

### Playing Phase

- After all players have bid, the playing phase begins
- The player with the highest bid goes first
- You must follow suit if possible
- Click on a card in your hand to play it
- The AI players will automatically make their moves

### Scoring

- **Exact bid:** Score = 11 × bid + 10
- **Wrong bid:** Score = -10 × |bid - actual tricks won|

## GUI Features

### Left Panel - Game Information

- **Status:** Shows whose turn it is
- **Round Info:** Current round number, trump suit, and cards per hand
- **Bidding:** Shows all players' bids
- **Tricks Won:** Shows how many tricks each player has won
- **Current Trick:** Shows the cards played in the current trick
- **Controls:** New Game and Load Model buttons

### Center Panel - Your Hand

- Displays your cards as clickable buttons
- Cards are arranged in a grid layout
- Only legal cards are clickable during play

### Right Panel - AI Players

- Shows how many cards each AI player has remaining
- Updates in real-time as cards are played

### Action Panel

- During bidding: Shows numbered buttons for making bids
- During playing: Shows your hand of cards

## Controls

- **New Game:** Start a new game with the same settings
- **Load Model:** Open a file dialog to load a different trained model
- **Card Buttons:** Click to play a card (only during your turn)
- **Bid Buttons:** Click to make a bid (only during bidding phase)

## Game Rules Reminder

### Judgement Rules

1. **Rounds:** Card counts vary (7, 6, 5, ..., 2, then 3, 4, ..., 7)
2. **Trump Suits:** Rotate through No Trump, Spades, Diamonds, Clubs, Hearts
3. **Bidding:** Total bids must not equal the number of players (4)
4. **Playing:** Must follow suit if possible
5. **Trick Resolution:** Highest trump wins, or highest card of led suit if no trump

### Valid Moves

- **Bidding:** Any number from 0 to the number of cards in hand
- **Playing:** Any card in your hand that follows suit rules

## Troubleshooting

### Common Issues

1. **"No trained model found"**

   - Make sure you have a trained model in the `models/` directory
   - Use the "Load Model" button to select a different model file

2. **"Invalid Bid" warning**

   - The total of all bids cannot equal 4 (the number of players)
   - Choose a different bid value

3. **"Invalid Card" warning**

   - You must follow suit if you have cards of the led suit
   - Choose a card that follows the suit rules

4. **GUI not responding**
   - The AI players take a moment to think
   - Wait for the AI moves to complete
   - If stuck, try clicking "New Game"

### Performance Tips

- The GUI works best with a trained model
- Without a model, the AI plays randomly
- Larger models may take longer to load and respond

## File Structure

```
judgement-rl/
├── gui_interface.py      # Main GUI implementation
├── play_against_ai.py    # Simple launcher script
├── GUI_README.md         # This file
├── models/               # Directory for trained models
│   └── selfplay_best_agent.pth  # Default model to load
├── judgement_env.py      # Game environment
├── agent.py             # AI agent implementation
└── state_encoder.py     # State encoding utilities
```

## Customization

You can modify the GUI by editing `gui_interface.py`:

- **Window size:** Change `self.root.geometry("1200x800")`
- **AI delay:** Adjust the delay in `self.root.after(500, self.game_loop)`
- **Colors and styling:** Modify the ttk widgets and styling
- **Game settings:** Change the number of players or max cards

## Requirements

- Python 3.7+
- tkinter (usually included with Python)
- torch
- numpy
- All other dependencies from `requirements.txt`

## Support

If you encounter issues:

1. Check that all dependencies are installed
2. Ensure you have a valid trained model
3. Try running with a fresh game
4. Check the console output for error messages
