"""Encoding module for converting game state to neural network inputs.

This module provides functions to encode the game state into tensor representations suitable
for the neural network: its Entities (Cards, Relics, Potions, Monsters, the Character, Event
Options and Rooms) as tokens, and its per-state fields such as the energy and the Map.
"""
