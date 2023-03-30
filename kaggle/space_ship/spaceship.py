import random

# CARD constant
SUIT_TUPLE = ('spades', 'hearts', 'clubs', 'diamonds')
RANK_TUPLE = ('Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King')

NCARDS = 8

def get_card(deck):
  return deck.pop() # pops one off the top of the deck and return it
