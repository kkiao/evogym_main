from src.body import make_fixed_body, mutate_body

parent = make_fixed_body()
child = mutate_body(parent)
print("親:\n", parent)
print("子:\n", child)