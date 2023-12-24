# fliptactoe-ai
making the game flip tac toe and an ai for it - may only need reinforcement learning for this one

--> This is likely more a journal than typical readme, so read at your own 'risk' lol<--

I just finished making the game: it can do 10,000 games in ~12 seconds on my machine.

I will be very curious what the AI figures out on this, becuase I have played this game enough with my family to think that I know a pretty optimized strategy. This game is likely "solved" (but just not yet by humans because it's not popular enough for someone to have gone through it) and will be very annoying to play against the AI, but that's the point.

This is the first game I feel like I made a good string representation for, so I'm pretty proud of that. I also tried to keep the `available_moves` optimized because that would take up a lot of time if I made it go throught the whole board for every person's turn to do that. That'll likely help a lot with training the AI. *Just did the time analysis for this, it saves me ~1 second for 30,000 games (38 to 36.9 seconds) doing these optimizations, so it's almost negligable. That's interesting and I'm gonna look into it more: I really think the optimizations should be significant*