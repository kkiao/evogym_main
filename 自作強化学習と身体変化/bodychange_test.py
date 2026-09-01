from src.body import make_climber_seed_body, mutate_body

parent = make_climber_seed_body()
child = mutate_body(parent)
print("親:\n", parent)
print("子:\n", child)
