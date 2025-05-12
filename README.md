# RL-tutorial

Quick research project to learn the basics of reinforcement learning using PyTorch. This is a 3-stage project.

## Stage 1 (POC research)
I am implementing different reinforcement learning agents on some of gymnasium's games.
- Random agent on CartPole
- DQNAgent on CartPole

## Stage 2 (student building)

The goal of this stage is to optimize elevator routes in student buildings.
A 25 floor apartment building, with 2 elevators that can travel 1 floor every 2 seconds, but if it stops on a floor it will stop for 10 seconds.
The 2nd floor is mostly used for the gym, and the 3rd floor is mostly used for its study rooms. Everything else is living spaces, with anywhere between 15 to 40 people living per floor. At different times of the day, we expect people to:
1. leave their apartment to go to the gym/study space
2. leave the gym/study space to go back up to their apartment
3. leave their apartment to leave the building
4. leave the gym/study space to leave the building
5. enter the building to go the gym/study space
6. enter the building to go to their apartment
The weights of people's decisions will depend on the time of the day. A person isn't expected to make more than 1 decision per hour and must leave their apartment at some point in the day and return to their apartment before 1am. Each person must "sleep" anywhere between 6 to 10 hours a night (following a normal distribution, the time they start sleeping is also random between 10pm and 2am). People's decisions should be more likely to be taken 30 minutes past the hour (as we are simulating a mostly student populated building).

This stage's implementation is done in A2C.

## Stage 3 (campus building)

The goal of this stage is also to optimize elevator routes, this time for a campus building.
A 7 floor building, with 5 elevators that can travel 1 floor every 2 seconds, but if it stops on a floor it will stop for 10 seconds.
The first and second floor are mostly used for lounging and both serve as an entrance. The building's stairs are easy to access, and so students are less likely to travel up/down 1 floor using elevators.
Every half hour between 8:30am and 6pm, we expect people to travel from one floor to another. We assume upwards traffic from the first and second floor is more likely in the early hours of the morning & afternoon, whereas downwards traffic from upper floors to the first and second floor is more likely in the later hours of the morning & evening.

This stage's implementation is still to be determined